# Freight Rate Prediction

Predicts the posted rate of a truckload from its lane, equipment, weight, date and market index.
Trained on 48,000 loads (Jan-Oct 2025), scored on 12,000 loads (Nov-Dec 2025).

**Report:** [reports/report.pdf](reports/report.pdf) · **Step-by-step analysis log:** [docs/ANALYSIS_LOG.md](docs/ANALYSIS_LOG.md)

**Result:** on months the model never saw, it prices loads with **1.50% mean absolute percentage error ($36 on a ~$2,050 load)**, 9 in 10 loads within 3%. A rate-card baseline scores 4.15% ($96).

## Quick start

```bash
git clone <this-repo-url> && cd <repo-folder>
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the unit tests, trains the final model, predicts and runs the provided `score.py` (~1 min). Expected output:

- `validation_predictions.csv` - 12,000 rows, `load_id,predicted_rate`
- `data/december_chart_inputs.csv` - `predicted_rate` filled for the 31 December rows
- `scorer_results/candidate_december.png` - the December chart
- `reports/prediction_checks.md` - input coverage and prediction sanity checks
- console: `Validated 12,000 final predictions.` / `Validated 31 fixed December predictions.`

![December chart](scorer_results/candidate_december.png)

---

## Key findings

1. **Rates climb ~2-8% over the last month of every quarter** and reset when the next quarter opens (the quarter-end shipper push). Strongest for Flatbed (+7.9% by the last day), then Reefer (+5.0%) and Dry Van (+2.4%). December is a quarter end, so the December chart rises through the month.
2. **Price = distance x equipment x weight x market.** Rate per mile falls with distance (rate ~ distance^0.87); Reefer +13% and Flatbed +8% over Dry Van; ~+3% per 10k lb; a 10% higher market index adds ~1.5%.
3. **The market index is a daily signal with a weekly cycle** (Thursday peak, Sunday trough). Nov-Dec sits well below the spring peak (0.93 vs 1.30).
4. **1.4% of labels are corrupted** (rates at 0.16-0.45x or 2.1-5.2x of expected, with a clean gap in between). Training without removing them costs ~0.6 MAPE points.
5. **`quote_signal` looks useful but isn't**: its U-shaped relationship with rates is an artifact of short hauls, and disappears once distance is modelled properly.

## Data-quality issues

| Issue | Size | Handling |
|---|---|---|
| Corrupted `posted_rate` labels | 677 loads (1.41%) | flagged by ratio to a robust expected rate, removed from training ([reports/label_outliers.csv](reports/label_outliers.csv)) |
| Negative weights | 292 train / 145 validation | sign-flip errors: absolute value + flag (they price like their absolute value) |
| Weight capped at 47,500 lb | 1,191 loads | kept, flagged |
| Missing weight | 300 / 165 | training median + flag |
| Missing market index | 374 / 249 | that day's mean across all loads (it's a daily signal) |
| Synthetic, shifted coordinates | all cities | used only as relative location features |
| 8 cities only in validation | 12% of validation loads | no city/lane IDs in the trees; generalise through coordinates (tested: 1.9% vs 1.7% error on hidden cities) |
| December inputs have no market index / coordinates | 31 rows | coordinates from the city table; market index from that day's mean in `validation.csv` (features only, no labels) |

## Validation and split

`validation.csv` is the **two months after** the labelled data, so this is a forecasting problem and a random split would leak future market states. Every fold trains on an expanding window and tests on the next two months:

| Fold | Train | Test | Why |
|---|---|---|---|
| Q2 | Jan-Apr | May-Jun | same shape as the real task: train through month 1 of a quarter, predict months 2-3 |
| Q3 | Jan-Jul | Aug-Sep | same shape as the real task |
| X1 | Jan-Jun | Jul-Aug | stress test: predict into an unseen quarter |
| X2 | Jan-Aug | Sep-Oct | most recent data |

Boosting rounds are chosen by early stopping on the latest 20% of each training window (never the test months). Corrupted labels are removed from training; metrics are reported on clean test labels and on all test labels.

## Model

```
log(rate) =  market part (linear)            +  load part (LightGBM, linear trees)
             market index elasticity             distance, equipment, weight,
             quarter-end ramp x equipment        pickup/delivery coordinates,
             monthly drift                       circuity, weight flags
          +  lane premium (shrunk, out-of-fold)
```

- The **market part** is linear so it can extrapolate into Nov-Dec. The **trees never see a date or market value**: plain LightGBM used the daily market value as a hidden date ID and memorised day-level prices.
- **Linear trees** (a linear model in each leaf) fit smooth pricing curves better than flat-leaf trees: 1.62% -> 1.53%.
- 5 seeds averaged; a per-lane premium from out-of-fold residuals, shrunk by n/(n+10).

### Model comparison (MAPE, clean test labels)

| Model | Q2 | Q3 | X1 | X2 | Mean | Mean MAE |
|---|---|---|---|---|---|---|
| Rate-card baseline | 6.04% | 3.47% | 3.62% | 3.48% | 4.15% | $96.00 |
| Log-linear OLS + trend | 1.92% | 1.92% | 2.27% | 1.98% | 2.02% | $46.45 |
| LightGBM, time as a feature | 2.24% | 2.31% | 2.40% | 1.93% | 2.22% | $50.62 |
| Hybrid, standard trees | 1.54% | 1.47% | 1.94% | 1.52% | 1.62% | $37.47 |
| **Final: hybrid, linear trees, 5 seeds, lane premium** | **1.39%** | **1.35%** | **1.84%** | **1.42%** | **1.50%** | **$35.64** |

Full table: [reports/experiments.md](reports/experiments.md). Including the corrupted test labels, MAPE is ~3.9% for every reasonable model: those rates are 2-5x off and unpredictable by construction.

**Uncertainty:** 90% of loads fall within **-2.7% / +3.0%** of the prediction (calibrated on Q2+Q3, 89.8% coverage on the most recent months; lower at the start of an unseen quarter). See [reports/insights.md](reports/insights.md).

**How close to the ceiling:** a second model trained to predict the final model's errors from every load feature explains only ~3% of them, so the remaining error is close to the noise in negotiated spot prices.

## Business use

- **Quote quarter-end surcharges:** in the last 2-3 weeks of a quarter, Flatbed rates run up to ~$164 higher per median load, Reefer ~$110, Dry Van ~$46.
- **Price with a range, not a point:** the 90% band (-2.7% / +3.0%) gives a floor and ceiling for negotiation.
- **Data-quality guardrail:** the same ratio check that found the corrupted labels can flag suspicious rates as they are entered.
- **Lane monitoring:** lane premiums (-1.4% to +1.5%) show which lanes consistently price above or below their structure.

## Limitations

- No holiday calendar: the training months contain no holiday effect to learn, so Christmas week is priced at its market level.
- The December chart relies on the daily market index present in `validation.csv`; in production it would come from the market feed.
- The quarter-end ramp was learned from three quarters (Q1-Q3); a fourth (year-end) could behave differently. The last 1-3 days of past quarters were under-predicted by ~0.5-1%, so Dec 29-31 may be slightly low; a steeper tail term was tested but not adopted (gain within noise).
- Robustness checks (error by slice, `load_id` order, lane shocks, `quote_signal` variants, loss calibration, holidays) are in [docs/ANALYSIS_LOG.md](docs/ANALYSIS_LOG.md) section 6.7.

---

## How to run

Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
python -m pip install -r requirements.txt
python run_all.py                 # tests -> train -> predict -> score.py  (~2 min)
python run_all.py --full          # also EDA, model comparison, insights and the PDF report
```

Individual steps:

```bash
python scripts/eda.py             # figures + reports/eda_summary.md
python scripts/experiments.py     # model comparison on the time folds -> reports/experiments.md
python scripts/train.py           # final model -> artifacts/model.pkl
python scripts/predict.py         # validation_predictions.csv, fills the December file, reports/prediction_checks.md
python scripts/insights.py        # accuracy bands, 90% range, business numbers -> reports/insights.md
python scripts/build_report.py    # reports/report.pdf
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python -m unittest discover -s tests
```

## Repository layout

```
src/freight/
  config.py        paths and constants
  data.py          loading, cleaning, corrupted-label detection
  features.py      FeatureBuilder: geometry, weight, daily market, quarter calendar;
                   fills coordinates and market index for minimal inputs (December chart)
  models.py        baseline, log-linear, LightGBM variants, hybrid, lane premium, final_model()
  evaluation.py    time-based folds and metrics
scripts/           eda.py, experiments.py, train.py, predict.py, insights.py, build_report.py
tests/             unit tests (cleaning, features, fold design, models, output format)
reports/           report.pdf, figures, EDA summary, model comparison, insights, prediction checks, flagged labels
docs/              assignment brief, ANALYSIS_LOG.md (step-by-step record of the analysis)
validation_predictions.csv   final submission file
```
