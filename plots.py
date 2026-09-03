"""Figures for the summary PDF. Run after train_eval.py."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = np.load("output/model_outputs.npz")
test = pd.read_parquet("output/test_df.parquet").sort_values("target_date").reset_index(drop=True)
y, pred = d["y_test"], d["pred_blend"]

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# 1. Target distribution
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(np.clip(y, -0.5, 0.5), bins=60, color="#4477aa")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.mean(y), color="#cc3311", lw=1.2, ls="--", label=f"mean {np.mean(y):+.3f}")
ax.set(xlabel="open->close return", ylabel="count", title="2024 test: open-to-close return on offering days")
ax.legend()
fig.tight_layout()
fig.savefig("output/fig_target_hist.png", bbox_inches="tight")

# 2. Mean actual return by predicted decile
dec = pd.qcut(pred, 10, labels=False, duplicates="drop")
g = pd.DataFrame({"dec": dec, "y": y}).groupby("dec")["y"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(g.index, g["mean"], color=["#cc3311" if v < 0 else "#228833" for v in g["mean"]])
ax.set(xlabel="predicted-return decile (0 = most bearish)", ylabel="mean actual open->close",
       title="Monotonic signal: actual return by predicted decile (2024)")
fig.tight_layout()
fig.savefig("output/fig_decile.png", bbox_inches="tight")

# 3. Cumulative PnL: short bottom-quintile of predictions each day, equal weight
k = max(1, int(len(pred) * 0.2))
order = np.argsort(pred)
is_short = np.zeros(len(pred), bool)
is_short[order[:k]] = True
daily = pd.DataFrame({"date": test["target_date"], "pnl": np.where(is_short, -y, 0.0)})
daily = daily[is_short].groupby("date")["pnl"].mean()
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(daily.index, daily.cumsum(), color="#4477aa")
ax.set(xlabel="", ylabel=chr(931) + " daily mean short return (gross)",
       title=f"Edge stability: short bottom-20% predicted names at the open (2024)\n"
             f"{k} names over {daily.size} event-days, mean {daily.mean():+.3f}/name, no costs")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("output/fig_pnl.png", bbox_inches="tight")

print("wrote output/fig_*.png")
