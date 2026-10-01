"""Compare tree algorithms for the load part of the hybrid model (same market part,
features, folds and early stopping).

Run:  python scripts/compare_algorithms.py
Writes reports/algorithms.csv and reports/algorithms.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import xgboost as xgb

from freight import config, data
from freight.evaluation import FOLDS
from freight.features import FeatureBuilder
from freight.models import GBM, TIME_FEATURE, LogLinear


class XGBHybrid(GBM):
    """Same hybrid as GBM(trend="hybrid"), with XGBoost for the load part."""

    def __init__(self, xparams: dict):
        super().__init__(trend="hybrid", drop=("quote_signal",))
        self.xparams = xparams

    def fit(self, X, y):
        self.linear = LogLinear(use_trend=True).fit(X, y)
        target = np.log(np.asarray(y, float)) - self._offset(X)
        F = self._features(X)
        params = {"objective": "reg:squarederror", "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8,
                  "lambda": 1.0, "tree_method": "hist", "seed": config.RANDOM_SEED, "verbosity": 0, **self.xparams}

        def matrix(mask):
            return xgb.DMatrix(F[mask], label=target[mask], enable_categorical=True)

        early = (X[TIME_FEATURE] <= X[TIME_FEATURE].quantile(0.8)).to_numpy()
        probe = xgb.train(params, matrix(early), 5000, evals=[(matrix(~early), "valid")],
                          early_stopping_rounds=200, verbose_eval=False)
        self.rounds = int((probe.best_iteration + 1) * 1.15)
        self.model = xgb.train(params, matrix(np.ones(len(F), bool)), self.rounds)
        return self

    def predict(self, X):
        return np.exp(self.model.predict(xgb.DMatrix(self._features(X), enable_categorical=True)) + self._offset(X))


def candidates():
    lgb = lambda params, rounds=None: GBM(trend="hybrid", params=params, drop=("quote_signal",), num_rounds=rounds)
    return {
        "LightGBM, linear trees": lgb({"num_leaves": 31, "linear_tree": True}),
        "LightGBM, standard trees": lgb({"num_leaves": 31}),
        "XGBoost, 31 leaves": XGBHybrid({"grow_policy": "lossguide", "max_leaves": 31, "max_depth": 0}),
        "XGBoost, depth 6": XGBHybrid({"max_depth": 6}),
        "Extra-trees (LightGBM)": lgb({"num_leaves": 31, "extra_trees": True}, 4000),
        "Random forest (LightGBM)": lgb({"boosting": "rf", "num_leaves": 255, "bagging_fraction": 0.63,
                                         "bagging_freq": 1, "feature_fraction": 0.6, "min_data_in_leaf": 5}, 500),
    }


def main() -> None:
    train = data.flag_label_outliers(data.load_train())
    builder = FeatureBuilder().fit(train, data.load_validation())
    X, y = builder.transform(train), train[config.TARGET]
    clean = train["label_outlier"].eq(0)

    rows = []
    for name, model in candidates().items():
        for fold in FOLDS:
            tr, te = fold.split(train)
            model.fit(X[tr & clean], y[tr & clean])
            te = te & clean
            pred = model.predict(X[te])
            rows.append({"algorithm": name, "fold": fold.name.split(":")[0],
                         "MAPE": np.mean(np.abs(pred / y[te] - 1)), "MAE": np.mean(np.abs(pred - y[te]))})
        print(f"{name} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(config.REPORTS_DIR / "algorithms.csv", index=False)
    table = res.pivot_table(index="algorithm", columns="fold", values="MAPE", sort=False)
    table["Mean MAPE"] = table.mean(axis=1)
    table["Mean MAE"] = res.groupby("algorithm", sort=False)["MAE"].mean()
    lines = ["# Load-part algorithm comparison (same hybrid design, single seed, no lane premium)", "",
             "| Algorithm | " + " | ".join(table.columns) + " |", "|---" * (len(table.columns) + 1) + "|"]
    for name, r in table.iterrows():
        cells = [f"{v:.2%}" for v in r.iloc[:-1]] + [f"${r.iloc[-1]:.2f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (config.REPORTS_DIR / "algorithms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
