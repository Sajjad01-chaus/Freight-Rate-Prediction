"""Time-based validation folds and metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    test_start: str
    test_end: str

    def split(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        train = frame["date"] <= self.train_end
        test = (frame["date"] >= self.test_start) & (frame["date"] <= self.test_end)
        return train, test


# The real task is: train through October (month 1 of Q4), predict Nov-Dec.
# Q-folds copy that shape; X-folds test predicting into an unseen quarter.
FOLDS = (
    Fold("Q2: Jan-Apr -> May-Jun", "2025-04-30", "2025-05-01", "2025-06-30"),
    Fold("Q3: Jan-Jul -> Aug-Sep", "2025-07-31", "2025-08-01", "2025-09-30"),
    Fold("X1: Jan-Jun -> Jul-Aug", "2025-06-30", "2025-07-01", "2025-08-31"),
    Fold("X2: Jan-Aug -> Sep-Oct", "2025-08-31", "2025-09-01", "2025-10-31"),
)


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err) / y_true)),
        "MdAPE": float(np.median(np.abs(err) / y_true)),
        "Bias%": float(np.mean(y_pred / y_true - 1)),
    }
