"""python test_train_eval.py"""
import numpy as np
import pandas as pd

from train_eval import date_group_folds, directional_accuracy

# 1. date_group_folds: no date's rows ever split across a train/val boundary
dates = np.array(sorted(pd.to_datetime(
    ["2024-01-01"] * 11 + ["2024-01-02"] * 3 + ["2024-01-03"] * 5 + ["2024-01-04"] * 2
    + ["2024-01-05"] * 4 + ["2024-01-06"] * 6 + ["2024-01-07"] * 3 + ["2024-01-08"] * 4
)))
np.random.default_rng(0).shuffle(dates)
for tr, va in date_group_folds(dates, n_splits=4):
    assert set(dates[tr]) & set(dates[va]) == set(), "a date landed on both sides of a fold"
    assert len(tr) + len(va) <= len(dates)
print("date_group_folds: no date straddles a fold boundary -- ok")

# 2. directional_accuracy: this was the bug the reviewer flagged --
# sign(0) never matching sign(y_true) used to make "predict 0" score ~0
# instead of "no call made". Zero predictions must be excluded, not counted
# as misses.
y_true = np.array([0.05, -0.02, 0.01, -0.10, 0.03])
assert np.isnan(directional_accuracy(y_true, np.zeros_like(y_true))), \
    "an all-zero prediction should score NaN (no directional call), not ~0"

y_pred = np.array([0.01, -0.01, 0.01, 0.01, -0.01])  # matches sign on 3 of 5
assert directional_accuracy(y_true, y_pred) == 0.6

y_pred_mixed = np.array([0.01, -0.01, 0.0, -0.10, 0.03])  # one exact zero, excluded from denominator
assert directional_accuracy(y_true, y_pred_mixed) == 1.0  # 4/4 non-zero calls all correct
print("directional_accuracy: zero predictions excluded, non-zero predictions scored correctly -- ok")
