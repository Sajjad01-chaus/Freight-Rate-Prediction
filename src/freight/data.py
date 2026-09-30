"""Loading and cleaning of the raw load files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_train() -> pd.DataFrame:
    return load_raw(config.TRAIN_PATH)


def load_validation() -> pd.DataFrame:
    return load_raw(config.VALIDATION_PATH)


def city_table(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for frame in frames:
        for role in ("pickup", "delivery"):
            parts.append(frame[[role, f"{role}_lat", f"{role}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    cities = pd.concat(parts).drop_duplicates()
    if cities["city"].duplicated().any():
        raise ValueError("a city maps to more than one coordinate pair")
    return cities.set_index("city").sort_index()


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["weight_was_missing"] = out["weight"].isna().astype(np.int8)
    out["weight_was_negative"] = (out["weight"] < 0).astype(np.int8)
    out["weight"] = out["weight"].abs()  # negative weights are sign-flip errors
    out["weight_at_cap"] = (out["weight"] >= config.WEIGHT_CAP).astype(np.int8)
    out["market_index_was_missing"] = out["market_index"].isna().astype(np.int8)
    return out


def expected_rate_per_mile(frame: pd.DataFrame) -> pd.Series:
    rpm = frame[config.TARGET] / frame["distance"]
    band = pd.qcut(frame["distance"], 20, labels=False, duplicates="drop")
    return rpm.groupby([frame["equipment"], band, frame["date"].dt.month]).transform("median")


def flag_label_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rate_ratio"] = out[config.TARGET] / out["distance"] / expected_rate_per_mile(out)
    out["label_outlier"] = (
        (out["rate_ratio"] < config.OUTLIER_LOW_RATIO) | (out["rate_ratio"] > config.OUTLIER_HIGH_RATIO)
    ).astype(np.int8)
    return out
