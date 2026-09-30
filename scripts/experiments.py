"""Model comparison on time-based folds.

Run:  python scripts/experiments.py
Writes reports/experiments.csv and reports/experiments.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from freight import config, data
from freight.evaluation import FOLDS, metrics
from freight.features import FeatureBuilder
from freight.models import GBM, LogBlend, LogLinear, RatePerMileBaseline, final_model


def named(model, name):
    model.name = name
    return model


def candidates():
    return [
        (RatePerMileBaseline(), True),
        (LogLinear(use_trend=False), True),
        (LogLinear(use_trend=True), True),
        (GBM(trend="none"), True),
        (GBM(trend="feature"), True),
        (GBM(trend="detrend"), True),
        (GBM(trend="detrend"), False),  # without label cleaning
        (GBM(trend="hybrid"), True),
        (named(GBM(trend="hybrid", params={"num_leaves": 31, "linear_tree": True}, drop=("quote_signal",)),
               "Hybrid, linear trees (single model)"), True),
        (final_model(), True),
        (LogBlend(GBM(trend="detrend"), GBM(trend="feature")), True),
        (LogBlend(GBM(trend="detrend"), LogLinear(use_trend=True)), True),
    ]


def main() -> None:
    train = data.flag_label_outliers(data.load_train())
    val = data.load_validation()
    builder = FeatureBuilder().fit(train, val)
    X, y = builder.transform(train), train[config.TARGET]
    clean_label = train["label_outlier"].eq(0)

    rows = []
    for fold in FOLDS:
        tr_mask, te_mask = fold.split(train)
        for model, drop_outliers in candidates():
            fit_mask = tr_mask & clean_label if drop_outliers else tr_mask
            label = model.name + ("" if drop_outliers else " [no label cleaning]")
            t0 = time.time()
            model.fit(X[fit_mask], y[fit_mask])
            pred = pd.Series(model.predict(X[te_mask]), index=X.index[te_mask])
            for scope, mask in (("clean", te_mask & clean_label), ("all", te_mask)):
                m = metrics(y[mask], pred[mask])
                rows.append({"fold": fold.name, "model": label, "test_rows": scope, **m,
                             "seconds": round(time.time() - t0, 1)})
            print(f"{fold.name:26s} {label:52s} MAPE(clean)={rows[-2]['MAPE']:.4f}")

    res = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    res.to_csv(config.REPORTS_DIR / "experiments.csv", index=False)

    order = list(dict.fromkeys(res["model"]))
    summary = (res.groupby(["test_rows", "model"])[["MAE", "RMSE", "MAPE", "MdAPE", "Bias%"]].mean()
               .reset_index())
    holdout = res[res["fold"] == FOLDS[-1].name].set_index(["test_rows", "model"])
    lines = ["# Model comparison (time-based folds)", ""]
    for scope in ("clean", "all"):
        lines += [f"## Test rows: {scope}", "",
                  "| Model | Mean MAE | Mean MAPE | Mean RMSE | Holdout F3 MAE | Holdout F3 MAPE | F3 Bias |",
                  "|---|---|---|---|---|---|---|"]
        s = summary[summary["test_rows"] == scope].set_index("model").loc[order]
        for name, r in s.iterrows():
            h = holdout.loc[(scope, name)]
            lines.append(f"| {name} | {r.MAE:.2f} | {r.MAPE:.2%} | {r.RMSE:.2f} | {h.MAE:.2f} | {h.MAPE:.2%} | "
                         f"{h['Bias%']:+.2%} |")
        lines.append("")
    (config.REPORTS_DIR / "experiments.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
