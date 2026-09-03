import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

SEED = 0
TFIDF_MAX_FEATURES = 500

PRE_OPEN_COLS = [
    "log_prior_close", "discount_pct", "has_discount", "log_dollar_amount",
    "n_news", "hours_before_open", "headline_len", "body_len", "dow",
    "offering_seq", "has_prior_offering", "symbol_avg_prior_ret", "days_since_last_offering",
    "vol_20d", "mom_5d",
]
NUMERIC_COLS = ["gap_pct"] + PRE_OPEN_COLS
CLIP_COLS = ["vol_20d", "mom_5d", "symbol_avg_prior_ret"]


def _prep_numeric(train_df, test_df, cols):
    tr = train_df[cols].astype(float).copy()
    te = test_df[cols].astype(float).copy()

    for c in CLIP_COLS:
        lo, hi = tr[c].quantile([0.01, 0.99])
        tr[c] = tr[c].clip(lo, hi)
        te[c] = te[c].clip(lo, hi)

    med = tr.median()
    return tr.fillna(med), te.fillna(med)


def build_feature_matrix(train_df, test_df, include_gap=True):
    keyword_cols = [c for c in train_df.columns if c.startswith("kw_")]
    base_cols = NUMERIC_COLS if include_gap else PRE_OPEN_COLS
    numeric_cols = base_cols + keyword_cols
    n_structured = len(numeric_cols)

    x_num_train, x_num_test = _prep_numeric(train_df, test_df, numeric_cols)
    scaler = StandardScaler()
    x_num_train = scaler.fit_transform(x_num_train)
    x_num_test = scaler.transform(x_num_test)

    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2), min_df=3, stop_words="english")
    x_txt_train = tfidf.fit_transform(train_df["text"])
    x_txt_test = tfidf.transform(test_df["text"])

    x_train = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(x_num_train), x_txt_train]))
    x_test = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(x_num_test), x_txt_test]))
    feature_names = numeric_cols + list(tfidf.get_feature_names_out())

    return x_train, x_test, feature_names, n_structured


def winsorized_target(train_df, lo_q=0.02, hi_q=0.98):
    lo, hi = train_df["target_ret"].quantile([lo_q, hi_q])
    return np.clip(train_df["target_ret"].to_numpy(dtype=float), lo, hi)
