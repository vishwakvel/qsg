"""Sanity checks that the tradable blend's out-of-sample signal isn't noise.
Run after train_eval.py: python robustness_check.py

1. Permutation test: shuffle the training labels, refit ridge (alpha fixed at
   the CV-selected value to keep 200 refits cheap), score against the real,
   unshuffled 2024 test set. This is the R2 achievable with no real
   train/test relationship -- if the real R2 sits far outside this null
   distribution, the signal isn't an artifact of the pipeline.
2. Block bootstrap by target_date: resample whole trading days (not
   individual rows) with replacement, since names sharing a target_date are
   not independent draws. Gives a 95% interval on test R2 and short-leg PnL.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from features import SEED, build_feature_matrix, winsorized_target
from train_eval import clip_pred, fit_predict, trading_eval

N_PERMUTATIONS = 200
N_BOOTSTRAP = 1000
RIDGE_ALPHA = 300  # the value best_ridge_alpha selects; fixed here so the permutation loop doesn't rerun CV 200 times


def permutation_null(x_train, x_test, y_train, y_test, n=N_PERMUTATIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    null_r2 = np.empty(n)
    for i in range(n):
        y_shuf = rng.permutation(y_train)
        m = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(x_train, y_shuf)
        null_r2[i] = r2_score(y_test, clip_pred(m.predict(x_test)))
    return null_r2


def block_bootstrap_by_date(dates, y_true, pred, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(dates)
    idx_by_date = {d: np.where(dates == d)[0] for d in uniq}
    r2s, shorts = np.empty(n), np.empty(n)
    for i in range(n):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_date[d] for d in sampled])
        yt, pb = y_true[rows], pred[rows]
        r2s[i] = r2_score(yt, pb)
        shorts[i] = trading_eval(yt, pb)["short_leg_mean_pnl"]
    return r2s, shorts


def main():
    df = pd.read_parquet("output/dataset.parquet")
    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("target_date").reset_index(drop=True)
    train_df = df[df["target_date"] < "2024-01-01"].reset_index(drop=True)
    test_df = df[df["target_date"] >= "2024-01-01"].reset_index(drop=True)
    y_test = test_df["target_ret"].to_numpy(dtype=float)

    x_train, x_test, _, _ = build_feature_matrix(train_df, test_df, include_gap=False)
    y_train = winsorized_target(train_df)

    real_ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(x_train, y_train)
    real_r2 = r2_score(y_test, clip_pred(real_ridge.predict(x_test)))
    null_r2 = permutation_null(x_train, x_test, y_train, y_test)

    preds = fit_predict(train_df, test_df, include_gap=False)
    pred_blend = preds["blend"]
    boot_r2, boot_short = block_bootstrap_by_date(test_df["target_date"].to_numpy(), y_test, pred_blend)

    out = {
        "permutation_test": {
            "note": "200 ridge refits on label-shuffled training targets (alpha fixed "
                    "at the CV-selected value), each scored against the real 2024 test "
                    "set -- the null distribution of R2 achievable with no genuine "
                    "train/test relationship.",
            "real_ridge_test_r2": float(real_r2),
            "null_mean": float(null_r2.mean()),
            "null_std": float(null_r2.std()),
            "null_max": float(null_r2.max()),
            "real_exceeds_null_pct": float(100 * np.mean(real_r2 > null_r2)),
        },
        "block_bootstrap_by_date": {
            "note": "1000 resamples of whole trading days (not individual rows) with "
                    "replacement, tradable blend predictions, 95% percentile interval.",
            "r2_point": float(r2_score(y_test, pred_blend)),
            "r2_ci95": [float(np.percentile(boot_r2, 2.5)), float(np.percentile(boot_r2, 97.5))],
            "r2_pct_le_zero": float(np.mean(boot_r2 <= 0)),
            "short_leg_pnl_point": float(trading_eval(y_test, pred_blend)["short_leg_mean_pnl"]),
            "short_leg_pnl_ci95": [float(np.percentile(boot_short, 2.5)), float(np.percentile(boot_short, 97.5))],
            "short_leg_pnl_pct_le_zero": float(np.mean(boot_short <= 0)),
        },
    }
    print(json.dumps(out, indent=2))
    with open("output/robustness.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
