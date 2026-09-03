import bisect
import html
import re
from collections import defaultdict

import numpy as np
import pandas as pd

YEARS = [2021, 2022, 2023, 2024]

def load_offerings():
    dfs = [pd.read_csv(f"temp_offerings_{y}_anon.tsv", sep="\t") for y in YEARS]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["headline"] = df["headline"].fillna("").apply(html.unescape)
    df["body"] = df["body"].fillna("").apply(html.unescape)
    df["text"] = df["headline"] + ". " + df["body"]
    return df


def load_prices():
    df = pd.read_csv("temp_prices_2021_2024_anon.tsv", sep="\t")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def target_trading_day(ts, trading_days):
    hour = ts.hour + ts.minute / 60
    cand = ts.normalize() if hour < 9.5 else ts.normalize() + pd.Timedelta(days=1)
    idx = bisect.bisect_left(trading_days, cand)
    return trading_days[idx] if idx < len(trading_days) else pd.NaT


KEYWORDS = [
    "registered direct", "private placement", "public offering", "secondary offering",
    "follow on", "at the market", "shelf", "warrant", "convertible", "preferred",
    "common stock", "units", "upsiz", "price", "terminat", "dilut", "bought deal",
    "institutional investor", "reverse split", "ipo", "pre funded"
]
DOLLAR_RE = re.compile(r"\$\s?([\d,.]+)\s*(million|billion|m\b|b\b)?", re.IGNORECASE)
_UNIT = r"(?:common\s+)?(?:shares?|units?|ADSs?|ADRs?)"
PRICE_PS_PATTERNS = [
    re.compile(rf"(?:offering price|purchase price|priced at|at a price of)\s*(?:of\s*)?\$([\d,]+\.?\d*)\s*per\s*{_UNIT}", re.IGNORECASE),
    re.compile(rf"\$([\d,]+\.?\d*)\s*per\s*{_UNIT}", re.IGNORECASE),
    re.compile(rf"\$([\d,]+\.?\d*)/{_UNIT}", re.IGNORECASE),
]
PRICE_PS_EXCLUDE_WORDS = ["exercise", "par value", "depositary", "preferred", "debenture", "note ", "bond"]
PLAUSIBLE_DISCOUNT_RANGE = (-0.9, 1.0)


def extract_dollar_amount(text):
    # Returns the largest plausible offering size in $millions.
    # "$7 million" -> 7 ; "$5,000,000" -> 5 ; bare small figures like a
    # "$1,057.21" exercise price are ignored (neither a unit word nor a
    # raw-dollar magnitude), which otherwise parsed as $1057M of noise.
    best = 0.0

    for amt, unit in DOLLAR_RE.findall(text):
        try:
            v = float(amt.replace(",", ""))
        except ValueError:
            continue
        unit = unit.lower()
        if unit.startswith("b"):
            v *= 1000.0
        elif unit.startswith("m"):
            pass
        elif v >= 100_000:
            v /= 1e6  # bare figure written out in dollars
        else:
            continue

        best = max(best, v)

    return best if best <= 5000 else 0.0  # >$5B is a parse artifact for these names


def extract_price_per_share(text):
    low = text.lower()

    for pat in PRICE_PS_PATTERNS:
        for m in pat.finditer(low):
            try:
                val = float(m.group(1).replace(",", ""))
            except ValueError:
                continue

            window = low[max(0, m.start() - 40):m.start()]

            if any(w in window for w in PRICE_PS_EXCLUDE_WORDS) or "warrant" in window[-15:]:
                continue
            if 0.005 < val < 5000:
                return val
            
    return np.nan


def build():
    offs = load_offerings()
    prices = load_prices()
    trading_days = sorted(prices["date"].unique())
    offs["target_date"] = [target_trading_day(t, trading_days) for t in offs["timestamp"]]
    offs = offs.dropna(subset=["target_date"])
    offs["price_ps"] = offs["text"].apply(extract_price_per_share)
    offs["dollar_amount_m"] = offs["text"].apply(extract_dollar_amount)

    def agg(g):
        g = g.sort_values("timestamp")

        return pd.Series({
            "n_news": len(g),
            "text": "\n".join(g["text"]),
            "headline": " || ".join(g["headline"]),
            "last_news_time": g["timestamp"].max(),
            "price_ps": g["price_ps"].dropna().iloc[-1] if g["price_ps"].notna().any() else np.nan,
            "dollar_amount_m": g["dollar_amount_m"].max()
        })

    grouped = offs.groupby(["symbol", "target_date"], as_index=False).apply(agg)
    prices["prior_close"] = prices.groupby("symbol")["close"].shift(1)
    px_by_symbol = {s: g.set_index("date") for s, g in prices.groupby("symbol")}
    rows = []
    prior_rets = defaultdict(list)
    last_offering_date = {}

    for r in grouped.to_dict("records"):
        symbol, target_date = r["symbol"], r["target_date"]
        sym_px = px_by_symbol.get(symbol)

        if sym_px is None or target_date not in sym_px.index:
            continue

        today = sym_px.loc[target_date]

        if pd.isna(today["prior_close"]):
            continue

        prior_close = float(today["prior_close"])
        open_, close_ = float(today["open"]), float(today["close"])

        if open_ <= 0 or prior_close <= 0:
            continue

        price_ps = r["price_ps"]
        discount_pct = (price_ps / prior_close - 1) if pd.notna(price_ps) else np.nan

        if pd.notna(discount_pct) and not (PLAUSIBLE_DISCOUNT_RANGE[0] <= discount_pct <= PLAUSIBLE_DISCOUNT_RANGE[1]):
            discount_pct = np.nan

        hist_rets = prior_rets[symbol]
        offering_seq = len(hist_rets)
        symbol_avg_prior_ret = float(np.mean(hist_rets)) if hist_rets else np.nan
        last_date = last_offering_date.get(symbol)
        days_since_last_offering = (target_date - last_date).days if last_date is not None else np.nan

        pre_px = sym_px[sym_px.index < target_date]["close"]
        pre_px = pre_px[pre_px > 0]
        vol_20d = pre_px.pct_change().tail(20).std()
        mom_5d = pre_px.iloc[-1] / pre_px.iloc[-6] - 1 if len(pre_px) >= 6 else np.nan

        target_ret = (close_ - open_) / open_

        rows.append({
            "symbol": symbol,
            "target_date": target_date,
            "n_news": r["n_news"],
            "text": r["text"],
            "headline": r["headline"],
            "hours_before_open": (pd.Timestamp(target_date) + pd.Timedelta(hours=9, minutes=30) - r["last_news_time"]).total_seconds() / 3600,
            "prior_close": prior_close,
            "gap_pct": (open_ - prior_close) / prior_close,
            "target_ret": target_ret,
            "discount_pct": discount_pct,
            "dollar_amount_m": r["dollar_amount_m"],
            "offering_seq": offering_seq,
            "symbol_avg_prior_ret": symbol_avg_prior_ret,
            "days_since_last_offering": days_since_last_offering,
            "vol_20d": vol_20d,
            "mom_5d": mom_5d,
        })

        prior_rets[symbol].append(target_ret)
        last_offering_date[symbol] = target_date

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "target_date"])
    df = df.sort_values("target_date").reset_index(drop=True)
    text_norm = df["text"].str.lower().str.replace("-", " ", regex=False)

    for kw in KEYWORDS:
        col = "kw_" + kw.replace(" ", "_")
        df[col] = text_norm.str.contains(kw, regex=False).astype(int)

    df["has_discount"] = df["discount_pct"].notna().astype(int)
    df["discount_pct"] = df["discount_pct"].fillna(0.0)
    df["log_dollar_amount"] = np.log1p(df["dollar_amount_m"])
    df["log_prior_close"] = np.log(df["prior_close"])
    df["headline_len"] = df["headline"].str.len()
    df["body_len"] = df["text"].str.len()
    df["dow"] = pd.to_datetime(df["target_date"]).dt.dayofweek
    df["has_prior_offering"] = (df["offering_seq"] > 0).astype(int)
    df["symbol_avg_prior_ret"] = df["symbol_avg_prior_ret"].fillna(0.0)
    df["days_since_last_offering"] = df["days_since_last_offering"].fillna(-1.0)
    df["mom_5d"] = df["mom_5d"].fillna(0.0)
    return df


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(df["target_date"].min(), df["target_date"].max())
    print(df["target_ret"].describe())
    print("rows with discount extracted:", df["has_discount"].sum())
    df.to_parquet("output/dataset.parquet")
