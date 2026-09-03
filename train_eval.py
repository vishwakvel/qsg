import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from features import SEED, build_feature_matrix, winsorized_target

PRED_CLIP = 0.75

def clip_pred(p):
    return np.clip(p, -PRED_CLIP, PRED_CLIP)


def directional_accuracy(y_true, y_pred):
    called = np.sign(y_pred) != 0
    if not called.any():
        return float("nan")
    return float(np.mean(np.sign(y_true[called]) == np.sign(y_pred[called])))


def date_group_folds(dates, n_splits=4):
    uniq = np.sort(pd.unique(dates))
    for tr_d, va_d in TimeSeriesSplit(n_splits=n_splits).split(uniq):
        tr_dates, va_dates = set(uniq[tr_d]), set(uniq[va_d])
        tr_rows = np.flatnonzero(pd.Series(dates).isin(tr_dates).to_numpy())
        va_rows = np.flatnonzero(pd.Series(dates).isin(va_dates).to_numpy())
        yield tr_rows, va_rows


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
    }


def best_ridge_alpha(train_df, include_gap=True):
    dates = train_df["target_date"].to_numpy()
    alphas = [3, 10, 30, 100, 300]
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        scores = []
        for tr, va in date_group_folds(dates):
            tr_df, va_df = train_df.iloc[tr], train_df.iloc[va]
            x_tr, x_va, _, _ = build_feature_matrix(tr_df, va_df, include_gap=include_gap)
            y_tr = winsorized_target(tr_df)
            y_va = va_df["target_ret"].to_numpy(dtype=float)
            m = Ridge(alpha=alpha, random_state=SEED).fit(x_tr, y_tr)
            scores.append(r2_score(y_va, clip_pred(m.predict(x_va))))
        if np.mean(scores) > best_score:
            best_score, best_alpha = np.mean(scores), alpha
    return best_alpha


def fit_predict(train_df, test_df, include_gap=True):
    x_train, x_test, feature_names, n_struct = build_feature_matrix(train_df, test_df, include_gap=include_gap)
    xs_train, xs_test = x_train[:, :n_struct].toarray(), x_test[:, :n_struct].toarray()

    y_train = winsorized_target(train_df)
    y_train_raw = train_df["target_ret"].to_numpy(dtype=float)

    alpha = best_ridge_alpha(train_df, include_gap=include_gap)
    ridge = Ridge(alpha=alpha, random_state=SEED).fit(x_train, y_train)
    pred_ridge = clip_pred(ridge.predict(x_test))
    rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=5, random_state=SEED, n_jobs=1).fit(xs_train, y_train)
    pred_rf = clip_pred(rf.predict(xs_test))
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED).fit(xs_train, y_train)
    pred_gb = clip_pred(gb.predict(xs_test))
    pred_blend = 0.5 * pred_ridge + 0.5 * pred_gb

    logit = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED, class_weight="balanced")
    logit.fit(x_train, (y_train_raw > 0).astype(int))
    proba = logit.predict_proba(x_test)[:, 1]

    return {
        "ridge": pred_ridge, "random_forest": pred_rf, "gradient_boosting": pred_gb,
        "blend": pred_blend, "proba_logit": proba, "alpha": alpha,
        "ridge_model": ridge, "feature_names": feature_names,
    }


def trading_eval(y_true, pred, q=0.2):
    k = max(1, int(len(y_true) * q))
    order = np.argsort(pred)
    short_ret = -y_true[order[:k]]
    long_ret = y_true[order[-k:]]
    return {
        "quantile": q,
        "n_per_leg": k,
        "short_leg_mean_pnl": float(short_ret.mean()),
        "short_leg_hit_rate": float((short_ret > 0).mean()),
        "long_leg_mean_pnl": float(long_ret.mean()),
        "long_short_mean_pnl": float(0.5 * (short_ret.mean() + long_ret.mean())),
        "all_names_sign_pnl": float((-np.sign(pred) * y_true).mean()),
    }


def main():
    results = {}

    df = pd.read_parquet("output/dataset.parquet")
    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("target_date").reset_index(drop=True)

    train_df = df[df["target_date"] < "2024-01-01"].reset_index(drop=True)
    test_df = df[df["target_date"] >= "2024-01-01"].reset_index(drop=True)
    print(f"train={len(train_df)} test={len(test_df)}")

    y_train = winsorized_target(train_df)
    y_test = test_df["target_ret"].to_numpy(dtype=float)
    
    preds = fit_predict(train_df, test_df, include_gap=False)
    preds_oracle = fit_predict(train_df, test_df, include_gap=True)

    results["baseline_zero"] = metrics(y_test, np.zeros_like(y_test))
    results["baseline_train_mean"] = metrics(y_test, np.full_like(y_test, y_train.mean()))
    results["baseline_always_short_dir_acc"] = float((y_test < 0).mean())
    results["baseline_gap_continues_NOT_TRADABLE"] = metrics(y_test, test_df["gap_pct"].to_numpy(dtype=float))

    for name in ["ridge", "random_forest", "gradient_boosting", "blend"]:
        results[name] = metrics(y_test, preds[name])
        results[f"{name}_oracle_with_gap_pct_NOT_TRADABLE"] = metrics(y_test, preds_oracle[name])
    results["ridge"]["alpha"] = preds["alpha"]

    y_test_cls = (y_test > 0).astype(int)
    proba = preds["proba_logit"]
    results["logistic_direction"] = {
        "accuracy": float(np.mean((proba > 0.5) == y_test_cls)),
        "auc": roc_auc_score(y_test_cls, proba),
        "baseline_majority_acc": float(max(y_test_cls.mean(), 1 - y_test_cls.mean())),
    }
    results["trading_eval_blend_2024"] = trading_eval(y_test, preds["blend"])
    results["trading_eval_blend_2024_oracle_with_gap_pct_NOT_TRADABLE"] = trading_eval(y_test, preds_oracle["blend"])

    wf, pooled_true, pooled_pred = {}, [], []

    for year in [2023, 2024]:
        tr = df[df["target_date"] < f"{year}-01-01"].reset_index(drop=True)
        te = df[(df["target_date"] >= f"{year}-01-01") & (df["target_date"] < f"{year + 1}-01-01")].reset_index(drop=True)

        if len(tr) < 200 or len(te) < 50:
            continue

        p = fit_predict(tr, te, include_gap=False)["blend"]
        yt = te["target_ret"].to_numpy(dtype=float)
        wf[str(year)] = {**metrics(yt, p), "n_train": len(tr), "n_test": len(te)}
        pooled_true.append(yt)
        pooled_pred.append(p)

    pt, pp = np.concatenate(pooled_true), np.concatenate(pooled_pred)
    wf["pooled"] = {**metrics(pt, pp), "trading": trading_eval(pt, pp)}
    results["walk_forward_blend"] = wf

    print(json.dumps(results, indent=2))

    ridge_coefs = pd.Series(preds["ridge_model"].coef_, index=preds["feature_names"]).sort_values()

    np.savez("output/model_outputs.npz", y_test=y_test, pred_ridge=preds["ridge"],
             pred_rf=preds["random_forest"], pred_gb=preds["gradient_boosting"],
             pred_blend=preds["blend"], pred_blend_oracle=preds_oracle["blend"],
             pred_gap=test_df["gap_pct"].to_numpy(dtype=float),
             proba_logit=proba, y_test_cls=y_test_cls)

    with open("output/results.json", "w") as f:
        json.dump(results, f, indent=2)

    ridge_coefs.head(15).to_frame("coef").to_csv("output/top_neg_coefs.csv")
    ridge_coefs.tail(15).to_frame("coef").to_csv("output/top_pos_coefs.csv")
    train_df.to_parquet("output/train_df.parquet")
    test_df.to_parquet("output/test_df.parquet")


if __name__ == "__main__":
    main()
