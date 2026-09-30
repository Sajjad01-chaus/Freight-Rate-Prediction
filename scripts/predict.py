"""Score validation.csv and the December chart inputs with the trained model.

Run:  python scripts/predict.py
Writes validation_predictions.csv and fills data/december_chart_inputs.csv in place.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from freight import config, data


def check(pred: np.ndarray, label: str) -> None:
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise SystemExit(f"{label}: non-finite or non-positive predictions")


def main() -> None:
    with open(config.ARTIFACTS_DIR / "model.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    builder, model = bundle["builder"], bundle["model"]

    val = data.load_validation()
    val_pred = model.predict(builder.transform(val))
    check(val_pred, "validation")
    template = pd.read_csv(config.TEMPLATE_PATH)
    by_id = pd.Series(val_pred, index=val["load_id"])
    if set(template["load_id"]) != set(by_id.index):
        raise SystemExit("template load_ids do not match validation.csv")
    template["predicted_rate"] = template["load_id"].map(by_id).round(2)
    template[["load_id", "predicted_rate"]].to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(template):,} predictions -> {config.PREDICTIONS_PATH.name} "
          f"(median ${template.predicted_rate.median():,.0f})")

    december = pd.read_csv(config.DECEMBER_PATH)
    columns = list(december.columns)
    dec_pred = model.predict(builder.transform(december.assign(date=pd.to_datetime(december["date"]))))
    check(dec_pred, "december")
    december["predicted_rate"] = np.round(dec_pred, 2)
    december[columns].to_csv(config.DECEMBER_PATH, index=False)
    print(f"Filled {len(december)} December rows -> {config.DECEMBER_PATH.relative_to(config.ROOT)}")
    print(december[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
