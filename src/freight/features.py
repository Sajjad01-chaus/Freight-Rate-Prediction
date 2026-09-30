"""Feature engineering shared by training and inference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .data import city_table, clean_features

EARTH_RADIUS_MILES = 3958.8
TIME_ORIGIN = pd.Timestamp("2025-01-01")


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(h))


@dataclass
class FeatureBuilder:
    """Holds the lookups needed at inference time: city coordinates, the daily
    market index and fill values. Unlabelled frames passed to `fit` only add
    cities and market readings."""

    cities: pd.DataFrame = None
    daily_market: pd.Series = None
    weight_fill: float = None
    quote_fill: float = None

    def fit(self, train: pd.DataFrame, *unlabeled: pd.DataFrame) -> "FeatureBuilder":
        self.cities = city_table(train, *unlabeled)
        self.weight_fill = float(train["weight"].abs().median())
        self.quote_fill = float(train["quote_signal"].median())
        self.daily_market = pd.Series(dtype=float)
        for frame in (train, *unlabeled):
            self.update_market(frame)
        return self

    def update_market(self, frame: pd.DataFrame) -> None:
        day = frame.groupby(frame["date"].dt.normalize())["market_index"].mean().dropna()
        merged = day if self.daily_market.empty else pd.concat([self.daily_market, day])
        self.daily_market = merged[~merged.index.duplicated(keep="last")].sort_index()

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = clean_features(self._complete(df))
        X = pd.DataFrame(index=df.index)

        X["lane"] = df["pickup"] + ">" + df["delivery"]
        X["distance"] = df["distance"]
        X["log_distance"] = np.log(df["distance"])
        for col in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
            X[col] = df[col]
        straight = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
        X["circuity"] = df["distance"] / np.maximum(straight, 1.0)
        X["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
        X["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]

        X["equipment"] = pd.Categorical(df["equipment"], categories=config.EQUIPMENT_TYPES)
        X["weight"] = df["weight"].fillna(self.weight_fill)
        for col in ("weight_was_missing", "weight_was_negative", "weight_at_cap"):
            X[col] = df[col]

        day_mean = df["date"].map(self.daily_market)
        X["market_index_day"] = day_mean
        X["market_index"] = df["market_index"].fillna(day_mean)
        X["log_market_index"] = np.log(X["market_index"])
        X["market_index_was_missing"] = df["market_index_was_missing"]
        X["quote_signal"] = df["quote_signal"].fillna(self.quote_fill)

        date = df["date"]
        quarter_end = date.dt.to_period("Q").dt.end_time.dt.normalize()
        X["day_of_week"] = date.dt.dayofweek
        X["days_since_start"] = (date - TIME_ORIGIN).dt.days
        X["days_to_quarter_end"] = (quarter_end - date).dt.days
        X["month_in_quarter"] = (date.dt.month - 1) % 3
        # rates climb over the last ~month of each quarter
        X["quarter_end_ramp"] = np.clip(1 - X["days_to_quarter_end"] / 31.0, 0, 1)
        return X

    def _complete(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add columns that minimal inputs (the December chart file) don't carry."""
        for role in ("pickup", "delivery"):
            for axis in ("lat", "lon"):
                if f"{role}_{axis}" not in df:
                    df[f"{role}_{axis}"] = df[role].map(self.cities[axis])
            if df[f"{role}_lat"].isna().any():
                unknown = sorted(df.loc[df[f"{role}_lat"].isna(), role].unique())
                raise ValueError(f"unknown {role} cities: {unknown}")
        for col in ("market_index", "quote_signal"):
            if col not in df:
                df[col] = np.nan
        no_market = ~df["date"].isin(self.daily_market.index)
        if no_market.any():
            raise ValueError(f"no market data for {sorted(df.loc[no_market, 'date'].dt.date.unique())}")
        return df
