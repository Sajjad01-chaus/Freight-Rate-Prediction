"""Candidate models. Each exposes fit(X, y) / predict(X) on the dollar scale and
works on log(rate) internally, so multiplicative effects become additive."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config

TIME_FEATURE = "days_since_start"
ID_COLUMNS = ("lane",)

# features describing *when* a load moves, as opposed to *what* moves
MARKET_FEATURES = (
    "market_index", "market_index_day", "log_market_index", "market_index_was_missing",
    "day_of_week", "days_since_start", "days_to_quarter_end", "month_in_quarter", "quarter_end_ramp",
)


class RatePerMileBaseline:
    """Median rate per mile by equipment and distance band (a simple rate card)."""

    name = "Baseline: rate-card median"

    def fit(self, X, y):
        self.edges = np.quantile(X["distance"], np.linspace(0, 1, 21))
        rpm = y / X["distance"]
        self.table = rpm.groupby([X["equipment"].astype(str), self._band(X)]).median()
        self.fallback = rpm.median()
        return self

    def _band(self, X):
        return np.clip(np.searchsorted(self.edges, X["distance"], side="right") - 1, 0, len(self.edges) - 2)

    def predict(self, X):
        keys = pd.MultiIndex.from_arrays([X["equipment"].astype(str), self._band(X)])
        return self.table.reindex(keys).fillna(self.fallback).to_numpy() * X["distance"].to_numpy()


class LogLinear:
    """OLS on log(rate). Its market/calendar terms are reused as the offset in the hybrid GBM."""

    MARKET_TERMS = ("log_market_index", "log_market_index_day", "qe_ramp", "qe_ramp_flatbed",
                    "qe_ramp_reefer", "trend_months")

    def __init__(self, use_trend: bool = True):
        self.use_trend = use_trend
        self.name = "Log-linear OLS" + (" + trend" if use_trend else "")

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        ld = X["log_distance"]
        w = X["weight"] / 1e4
        flatbed = X["equipment"].astype(str).eq("Flatbed").astype(float)
        reefer = X["equipment"].astype(str).eq("Reefer").astype(float)
        ramp = X["quarter_end_ramp"]
        # no day-of-week dummies: they would soak up the market index's weekly cycle
        D = pd.DataFrame({
            "const": 1.0, "log_dist": ld, "log_dist2": ld ** 2, "log_dist3": ld ** 3,
            "flatbed": flatbed, "reefer": reefer,
            "weight": w, "weight2": w ** 2, "weight_at_cap": X["weight_at_cap"],
            "pickup_lat": X["pickup_lat"], "pickup_lon": X["pickup_lon"],
            "delivery_lat": X["delivery_lat"], "delivery_lon": X["delivery_lon"],
            "log_market_index": X["log_market_index"],
            "log_market_index_day": np.log(X["market_index_day"]),
            "qe_ramp": ramp, "qe_ramp_flatbed": ramp * flatbed, "qe_ramp_reefer": ramp * reefer,
        }, index=X.index)
        if self.use_trend:
            D["trend_months"] = X[TIME_FEATURE] / 30.0
        return D.astype(float)

    def fit(self, X, y):
        D = self.design(X)
        beta, *_ = np.linalg.lstsq(D.to_numpy(), np.log(np.asarray(y, float)), rcond=None)
        self.coef = pd.Series(beta, index=D.columns)
        return self

    def predict(self, X):
        return np.exp(self.design(X).to_numpy() @ self.coef.to_numpy())

    def market_terms(self, X) -> np.ndarray:
        cols = [c for c in self.MARKET_TERMS if c in self.coef.index]
        return self.design(X)[cols].to_numpy() @ self.coef[cols].to_numpy()

    @property
    def monthly_trend(self) -> float:
        return float(self.coef.get("trend_months", 0.0))


DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": config.RANDOM_SEED,
}


class GBM:
    """LightGBM on log(rate). `trend` controls how time is handled:

    none     - no time feature
    feature  - days_since_start as a feature (trees hold the last level flat)
    detrend  - remove a linear monthly trend from the target, add it back at predict time
    hybrid   - log(rate) = linear market/calendar terms + trees on load features only.
               The trees never see a date or market value, so they can't memorise days.
    """

    def __init__(self, trend: str = "hybrid", params: dict | None = None, num_rounds: int | None = None,
                 drop: tuple[str, ...] = (), n_seeds: int = 1):
        assert trend in {"none", "feature", "detrend", "hybrid"}
        self.trend, self.drop, self.n_seeds = trend, tuple(drop), n_seeds
        self.params = {**DEFAULT_LGB_PARAMS, **(params or {})}
        self.num_rounds = num_rounds
        self.name = f"LightGBM (trend={trend})" + (f" -{','.join(drop)}" if drop else "")

    def _features(self, X):
        excluded = set(self.drop) | set(ID_COLUMNS)
        if self.trend == "hybrid":
            excluded |= set(MARKET_FEATURES)
        elif self.trend != "feature":
            excluded.add(TIME_FEATURE)
        return X[[c for c in X.columns if c not in excluded]]

    def _offset(self, X):
        if self.trend == "detrend":
            return self.linear.monthly_trend * X[TIME_FEATURE].to_numpy() / 30.0
        if self.trend == "hybrid":
            return self.linear.market_terms(X)
        return 0.0

    def fit(self, X, y):
        if self.trend in {"detrend", "hybrid"}:
            self.linear = LogLinear(use_trend=True).fit(X, y)
        target = np.log(np.asarray(y, float)) - self._offset(X)
        F = self._features(X)
        self.rounds = self.num_rounds or self._pick_rounds(X, F, target)
        self.boosters = [
            lgb.train({**self.params, "seed": self.params["seed"] + i}, lgb.Dataset(F, target),
                      num_boost_round=self.rounds)
            for i in range(self.n_seeds)
        ]
        self.feature_names = list(F.columns)
        return self

    @property
    def booster(self):
        return self.boosters[0]

    def _pick_rounds(self, X, F, target) -> int:
        # early stopping on the latest 20% of the window, then refit on all of it
        cut = X[TIME_FEATURE].quantile(0.8)
        inner_tr, inner_va = (X[TIME_FEATURE] <= cut).to_numpy(), (X[TIME_FEATURE] > cut).to_numpy()
        booster = lgb.train(
            self.params, lgb.Dataset(F[inner_tr], target[inner_tr]), num_boost_round=5000,
            valid_sets=[lgb.Dataset(F[inner_va], target[inner_va])],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        self.best_iteration = booster.best_iteration
        return int(booster.best_iteration * 1.15)

    def predict(self, X):
        F = self._features(X)
        return np.exp(np.mean([b.predict(F) for b in self.boosters], axis=0) + self._offset(X))


class LaneCorrected:
    """Adds a shrunken per-lane premium on top of a base model.

    The premium is the mean out-of-fold log residual of each lane in the training
    window, shrunk towards 0 by n / (n + shrink). Out-of-fold residuals are used
    because in-sample residuals are already partly absorbed by the trees.
    """

    def __init__(self, make_base, shrink: float = 10.0, k: int = 5):
        self.make_base, self.shrink, self.k = make_base, shrink, k
        self.name = f"{make_base().name} + lane premium"

    def fit(self, X, y):
        self.base = self.make_base().fit(X, y)
        y = np.asarray(y, float)
        folds = np.random.default_rng(config.RANDOM_SEED).integers(0, self.k, len(X))
        oof = np.zeros(len(X))
        for j in range(self.k):
            part = self.make_base()
            part.num_rounds, part.n_seeds = self.base.rounds, 1
            part.fit(X[folds != j], y[folds != j])
            oof[folds == j] = np.log(y[folds == j]) - np.log(part.predict(X[folds == j]))
        stats = pd.DataFrame({"lane": X["lane"].to_numpy(), "r": oof}).groupby("lane")["r"].agg(["sum", "count"])
        self.premium = stats["sum"] / (stats["count"] + self.shrink)
        return self

    def predict(self, X):
        adj = X["lane"].map(self.premium).fillna(0.0).to_numpy()
        return self.base.predict(X) * np.exp(adj)


class LogBlend:
    """Geometric mean of several models."""

    def __init__(self, *members, name: str | None = None):
        self.members = members
        self.name = name or "Blend: " + " + ".join(m.name for m in members)

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.exp(np.mean([np.log(m.predict(X)) for m in self.members], axis=0))


def final_base_model() -> GBM:
    return GBM(trend="hybrid", params={"num_leaves": 31, "linear_tree": True}, drop=("quote_signal",), n_seeds=5)


def final_model() -> LaneCorrected:
    model = LaneCorrected(final_base_model, shrink=10.0)
    model.name = "FINAL: hybrid linear-tree x5 seeds + lane premium"
    return model
