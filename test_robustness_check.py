"""python test_robustness_check.py"""
import numpy as np
from robustness_check import block_bootstrap_by_date

# perfect predictions -> every bootstrap draw should score R2=1.0 regardless of
# which dates get resampled, and every date's rows must move together
dates = np.array(["2024-01-01"] * 3 + ["2024-01-02"] * 5 + ["2024-01-03"] * 2)
y = np.linspace(-0.1, 0.1, len(dates))
r2s, _ = block_bootstrap_by_date(dates, y, y.copy(), n=50, seed=0)
assert np.allclose(r2s, 1.0), f"expected all R2=1.0 for perfect predictions, got {r2s[:5]}"
print("block_bootstrap_by_date: perfect predictions score R2=1.0 on every resample -- ok")
