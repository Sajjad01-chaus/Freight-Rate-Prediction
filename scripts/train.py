"""Train the final model on all labelled data (Jan-Oct 2025) and save artifacts.

Run:  python scripts/train.py
Writes artifacts/model.pkl, artifacts/training_summary.json, reports/label_outliers.csv
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from freight import config, data
from freight.features import FeatureBuilder
from freight.models import final_model


def main() -> None:
    train = data.flag_label_outliers(data.load_train())
    val = data.load_validation()

    # validation.csv has no labels; it only supplies cities and Nov-Dec market readings
    builder = FeatureBuilder().fit(train, val)
    X, y = builder.transform(train), train[config.TARGET]
    keep = train["label_outlier"].eq(0)

    model = final_model().fit(X[keep], y[keep])

    in_sample = np.abs(model.predict(X[keep]) / y[keep] - 1)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(config.ARTIFACTS_DIR / "model.pkl", "wb") as fh:
        pickle.dump({"builder": builder, "model": model}, fh)
    gbm = model.base
    gbm.booster.save_model(str(config.ARTIFACTS_DIR / "lightgbm_booster.txt"))

    flagged = train.loc[~keep, ["load_id", "date", "pickup", "delivery", "equipment", "distance",
                                config.TARGET, "rate_ratio"]]
    flagged.to_csv(config.REPORTS_DIR / "label_outliers.csv", index=False)

    summary = {
        "model": model.name,
        "training_rows": int(keep.sum()),
        "label_outliers_removed": int((~keep).sum()),
        "boosting_rounds": int(gbm.rounds),
        "seeds": gbm.n_seeds,
        "tree_features": gbm.feature_names,
        "market_component_coefficients": gbm.linear.coef[list(gbm.linear.MARKET_TERMS)].round(5).to_dict(),
        "lanes_with_premium": int(model.premium.size),
        "lane_premium_range": [round(float(model.premium.min()), 4), round(float(model.premium.max()), 4)],
        "in_sample_MAPE": round(float(in_sample.mean()), 5),
    }
    (config.ARTIFACTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
