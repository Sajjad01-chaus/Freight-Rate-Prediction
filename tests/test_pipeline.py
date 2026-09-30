import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from freight import config, data
from freight.evaluation import FOLDS, metrics
from freight.features import FeatureBuilder
from freight.models import GBM, MARKET_FEATURES, LogLinear


def make_loads(n=600, seed=0):
    """Small synthetic frame with the same schema and pricing shape as the real data."""
    rng = np.random.default_rng(seed)
    cities = {"A": (35.0, -90.0), "B": (40.0, -85.0), "C": (32.0, -97.0), "D": (42.0, -75.0)}
    names = list(cities)
    pickup = rng.choice(names, n)
    delivery = np.array([rng.choice([c for c in names if c != p]) for p in pickup])
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    distance = rng.uniform(100, 2500, n)
    equipment = rng.choice(config.EQUIPMENT_TYPES, n)
    weight = rng.uniform(5_000, 45_000, n)
    market = 1 + 0.1 * np.sin(np.asarray(dates.dayofyear))
    rate = 5.0 * distance ** 0.87 * market ** 0.13 * np.where(equipment == "Reefer", 1.13, 1.0)
    rate *= np.exp(rng.normal(0, 0.02, n))
    return pd.DataFrame({
        "load_id": [f"TR-{i:06d}" for i in range(n)],
        "pickup": pickup, "delivery": delivery,
        "pickup_lat": [cities[c][0] for c in pickup], "pickup_lon": [cities[c][1] for c in pickup],
        "delivery_lat": [cities[c][0] for c in delivery], "delivery_lon": [cities[c][1] for c in delivery],
        "distance": distance, "equipment": equipment, "weight": weight, "date": dates,
        "market_index": market, "quote_signal": rng.normal(2, 0.2, n), "posted_rate": rate,
    })


class TestCleaning(unittest.TestCase):
    def test_negative_weight_is_flipped_and_flagged(self):
        df = make_loads(5)
        df.loc[0, "weight"] = -30_000
        df.loc[1, "weight"] = np.nan
        out = data.clean_features(df)
        self.assertEqual(out.loc[0, "weight"], 30_000)
        self.assertEqual(out.loc[0, "weight_was_negative"], 1)
        self.assertEqual(out.loc[1, "weight_was_missing"], 1)

    def test_corrupted_labels_are_flagged(self):
        df = make_loads(2_000)
        df.loc[0, "posted_rate"] *= 4
        df.loc[1, "posted_rate"] *= 0.25
        out = data.flag_label_outliers(df)
        self.assertEqual(out.loc[[0, 1], "label_outlier"].tolist(), [1, 1])
        self.assertLess(out["label_outlier"].mean(), 0.01)

    def test_city_table_rejects_conflicting_coordinates(self):
        df = make_loads(20)
        df.loc[df.pickup == "A", "pickup_lat"] = 99.0
        with self.assertRaises(ValueError):
            data.city_table(df)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.train = make_loads()
        self.builder = FeatureBuilder().fit(self.train)

    def test_minimal_input_like_december_chart(self):
        chart = pd.DataFrame({
            "pickup": "A", "delivery": "B", "distance": 360.0, "equipment": "Dry Van",
            "weight": 32_000.0, "date": pd.date_range("2025-03-01", periods=5),
        })
        X = self.builder.transform(chart)
        self.assertFalse(X.drop(columns=["lane"]).isna().any().any())
        self.assertTrue((X["pickup_lat"] == 35.0).all())
        expected = self.builder.daily_market.reindex(chart["date"]).to_numpy()
        np.testing.assert_allclose(X["market_index"], expected)

    def test_unknown_city_raises(self):
        chart = pd.DataFrame({"pickup": ["Nowhere"], "delivery": ["B"], "distance": [100.0],
                              "equipment": ["Dry Van"], "weight": [1.0], "date": [pd.Timestamp("2025-01-05")]})
        with self.assertRaises(ValueError):
            self.builder.transform(chart)

    def test_date_without_market_data_raises(self):
        chart = self.train.head(1).drop(columns="market_index").assign(date=pd.Timestamp("2030-01-01"))
        with self.assertRaises(ValueError):
            self.builder.transform(chart)

    def test_quarter_end_ramp(self):
        df = self.train.head(3).copy()
        df["date"] = pd.to_datetime(["2025-03-31", "2025-03-01", "2025-02-15"])
        builder = FeatureBuilder().fit(pd.concat([self.train, df]))
        ramp = builder.transform(df)["quarter_end_ramp"].tolist()
        self.assertEqual(ramp[0], 1.0)
        self.assertGreater(ramp[1], 0.0)
        self.assertEqual(ramp[2], 0.0)


class TestValidationDesign(unittest.TestCase):
    def test_folds_never_train_on_the_future(self):
        dates = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-10-31")})
        for fold in FOLDS:
            train, test = fold.split(dates)
            self.assertLess(dates.date[train].max(), dates.date[test].min())
            self.assertFalse((train & test).any())

    def test_metrics(self):
        m = metrics([100, 200], [110, 180])
        self.assertAlmostEqual(m["MAE"], 15.0)
        self.assertAlmostEqual(m["MAPE"], 0.10)


class TestModels(unittest.TestCase):
    def setUp(self):
        train = make_loads(1_500)
        self.X = FeatureBuilder().fit(train).transform(train)
        self.y = train["posted_rate"]

    def test_log_linear_fits_synthetic_pricing(self):
        model = LogLinear(use_trend=False).fit(self.X, self.y)
        pred = model.predict(self.X)
        self.assertLess(np.mean(np.abs(pred / self.y - 1)), 0.03)

    def test_hybrid_trees_never_see_market_or_date(self):
        model = GBM(trend="hybrid", num_rounds=50).fit(self.X, self.y)
        self.assertFalse(set(model.feature_names) & set(MARKET_FEATURES))
        self.assertNotIn("lane", model.feature_names)
        pred = model.predict(self.X)
        self.assertTrue(np.isfinite(pred).all() and (pred > 0).all())


@unittest.skipUnless(config.PREDICTIONS_PATH.exists(), "run scripts/predict.py first")
class TestSubmissionFiles(unittest.TestCase):
    def test_validation_predictions_format(self):
        pred = pd.read_csv(config.PREDICTIONS_PATH)
        template = pd.read_csv(config.TEMPLATE_PATH)
        self.assertEqual(list(pred.columns), ["load_id", "predicted_rate"])
        self.assertEqual(pred["load_id"].tolist(), template["load_id"].tolist())
        self.assertTrue((pred["predicted_rate"] > 0).all())

    def test_december_chart_filled(self):
        dec = pd.read_csv(config.DECEMBER_PATH)
        self.assertEqual(len(dec), 31)
        self.assertTrue((dec["predicted_rate"] > 0).all())


if __name__ == "__main__":
    unittest.main()
