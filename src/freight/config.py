from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
EQUIPMENT_TYPES = ("Dry Van", "Flatbed", "Reefer")
WEIGHT_CAP = 47_500.0

# rate / expected-rate cut-offs for corrupted labels; clean loads sit in 0.8-1.2x,
# corrupted ones in 0.16-0.45x and 2.1-5.2x
OUTLIER_LOW_RATIO = 0.6
OUTLIER_HIGH_RATIO = 1.6

RANDOM_SEED = 42
