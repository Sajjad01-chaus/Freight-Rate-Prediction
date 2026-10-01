# Analysis log

Chronological record of every check, what it showed, and what I decided because of it.
Numbers are from the actual runs (`scripts/eda.py`, `scripts/experiments.py`, `scripts/train.py`).

---

## 0. Understanding the task

- Labelled data: `train_test.csv`, 48,000 loads, **2025-01-01 to 2025-10-31**.
- Scoring data: `validation.csv`, 12,000 loads, **2025-11-01 to 2025-12-31**, no labels.
- December chart file: 31 rows, one lane (Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb), only the date changes.
- Read `score.py`: it only validates format (12,000 ids, positive numbers, 31 December dates, fixed lane) and draws the chart. The real metric is computed by Spotter and is not disclosed, so I report MAE, RMSE, MAPE and bias.

**Two things noticed immediately**
1. Scoring data is the two months *after* training ends -> this is a forecasting problem, a random split would be wrong.
2. The December file has **no `market_index`, no `quote_signal`, no coordinates** -> the model must still produce a sensible date-dependent prediction from minimal input.

---

## 1. Basic profiling

| Check | Result |
|---|---|
| Duplicate `load_id` / duplicate rows | none |
| Nulls in train | `weight` 300, `market_index` 374 |
| Nulls in validation | `weight` 165, `market_index` 249 |
| Equipment | Dry Van 27,202 / Reefer 12,045 / Flatbed 8,753 (same mix in validation) |
| Cities | 64 in train, 72 in validation |
| Target `posted_rate` | min $57, median $2,031, max $25,533 |
| Rate per mile | 0.33 to 14.1 $/mi, median 2.15 |

The extreme rate-per-mile values (0.33 and 14.1) were the first sign of bad labels.

---

## 2. Data-quality issues and how each was handled

### 2.1 Corrupted labels (the biggest one)
- **How found:** divided each rate per mile by the median for its equipment x distance band x month. Histogram of that ratio (`reports/figures/01_label_corruption.png`).
- **What it showed:** clean loads sit in a tight band of **0.8x to 1.2x**. Separately, two clusters: **0.16x-0.45x** (337 loads) and **2.1x-5.2x** (340 loads). Nothing in between, so the cut is unambiguous.
- **Total:** 677 loads = **1.41%**.
- **Decision:** thresholds 0.6x and 1.6x. Flagged loads are removed from training and listed in `reports/label_outliers.csv`.
- **Proof it matters:** same model trained with vs without cleaning: holdout MAPE **1.55% vs 2.20%**.

### 2.2 Negative weights
- 292 in train, 145 in validation. Range -5,000 to -47,500, same shape as the positive distribution.
- Checked: loads with negative weight price like their absolute value (median $2.16/mi vs $2.15/mi).
- **Decision:** sign-flip entry error -> take `abs()`, keep a `weight_was_negative` flag.

### 2.3 Weight cap and floor
- 1,191 loads exactly at **47,500 lb** (legal payload cap), small pile-up at 5,000 lb.
- **Decision:** keep the value, add `weight_at_cap` flag (those loads sit ~1.5% below what a smooth weight curve would predict).

### 2.4 Missing values
- Weight missing: pricing no different from the rest -> fill with the training median + flag.
- Market index missing: it is a **daily** signal (within-day std only 0.025, and the within-day variation is pure noise, no regional pattern) -> fill with that day's mean across all loads.

### 2.5 Synthetic coordinates
- Each city always has the same coordinates (checked: 0 cities with more than one pair), but they are shifted: Los Angeles at lat 28.6, Boston longitude clipped at -69.5, Laredo lat clipped at 25.5.
- Distance is ~1.18x the great-circle distance of these coordinates.
- **Decision:** use coordinates as relative location features only; never rely on real geography.

### 2.6 Cities that only appear in validation
- 8 cities: Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego.
- They touch **12.1%** of validation loads; 87.8% of validation loads are on a lane seen in training.
- **Decision:** no city-ID or lane-ID features; the model generalises through coordinates and distance.
- **Test:** hid 8 random cities from training (3 different random draws). Error on loads touching a hidden city: **1.85-2.05%** vs 1.68% for known cities. Mild degradation only.

---

## 3. What drives the price (EDA findings)

### 3.1 Distance
- Rate per mile falls with distance: ~$2.7/mi for short hauls to ~$1.8/mi for 2,500+ mi (Dry Van). Fixed costs spread over more miles.
- Log-log slope: rate ~ distance^0.87.

### 3.2 Equipment
- Median rate per mile: Dry Van $2.05, Flatbed $2.22, Reefer $2.31.
- Regression premium over Dry Van: Flatbed +8.3%, Reefer +12.9%. Holds at every distance band (`02_rate_per_mile_vs_distance.png`).

### 3.3 Weight
- About +3% per 10,000 lb.

### 3.4 Market index
- Same value (plus small noise) for every load on a given day -> a daily market signal.
- Daily mean correlates **0.71** with the daily rate level.
- Strong **weekly cycle**: Thursday peak ~1.16, Sunday trough ~0.96.
- Tested same-day vs 3/7/14/28/56-day rolling means, 7/14-day lags, exponential smoothing and weekday-adjusted versions. Same-day beats short windows and lags (corr 0.69). A 30-day exponential average correlates slightly higher (0.74), but only because it tracks the slow time trend; smoothed market features were later tested inside the model and did not generalise (section 6.5, lever B). **Same-day value kept.**
- Validation period sits much lower: mean 0.927 vs 1.083 in training (spring peak ~1.30).
- Elasticity: ~0.12-0.14 (a 10% higher index -> ~1.3% higher rate). Measured from within-week variation so it isn't confused with the slow trend.

### 3.5 quote_signal (a trap)
- Linear correlation with rate: ~0.1, looked useless.
- Plotted against the residual of a log-linear model: clear **U-shape of +/-3%**, looked important.
- Checked against a flexible model instead: effect shrinks to **<0.3%**. The U-shape came from short hauls, where quote_signal has much wider spread (std 0.51 vs 0.20) and the linear model got the distance curve wrong.
- Chart `07_quote_signal.png` shows both lines.
- **Decision:** dropped from the final model. Holdout average improved slightly (1.56% -> 1.52%), and the December inputs don't carry it anyway, so dropping it removes a train/serve mismatch.

### 3.6 Day of week, holidays
- Day-of-week effect on rates is tiny once market index is known (+/-0.2%).
- No spikes around July 4, Memorial Day or Labor Day.

---

## 4. The time pattern (the key finding)

### 4.1 First look: "6% drift" (partly wrong)
- A simple log-linear model with market index left monthly residuals going from -3.2% (Jan) to +2.4% (Oct). I first read this as a steady ~0.6%/month trend.

### 4.2 First correction
- Extrapolating that straight-line trend failed badly in the Jan-Jun -> Jul-Aug fold (**7.8%** error).
- Reason: January to June, the market index and time rose together, so a regression can't separate "the market went up" from "time passed". The trend estimate soaked up part of the market effect.
- Fix: estimated the market elasticity from **within-week** variation only (the weekly cycle has nothing to do with the slow trend) -> elasticity 0.116.

### 4.3 What the remaining pattern really is
Weekly averages of what's left after the market effect:
- flat for the first two months of each quarter,
- **ramps up during the last month of every quarter** (March, June, September),
- drops back when the next quarter opens.

Ramp size by days to quarter end (Q1 / Q2 / Q3 all agree):

| Days to quarter end | ~30 | ~15 | 0 |
|---|---|---|---|
| Uplift vs. quarter base | +1% | +2.5-3.5% | **+5%** |

- Differs by equipment: Flatbed +4.2%, Reefer +2.8%, Dry Van +1.5% (average over the last month).
- Same for every distance band.
- Base level also steps up ~1.8% per quarter.
- **Business reading:** this is the quarter-end shipper push, shippers moving volume to hit quarterly numbers, which tightens capacity and lifts spot rates.
- **Why it matters for this task:** December is a quarter end and year end, so rates should climb through December.

### 4.4 Features added
- `days_to_quarter_end`, `month_in_quarter`, `quarter_end_ramp` (0 until ~31 days before quarter end, rising to 1 on the last day).
- Effect: best LightGBM holdout error went from 1.72% to 1.55%, average across folds from 3.78% to 2.03%.

---

## 5. Validation design

- **No random split.** Scoring data is the future, so every fold trains on an expanding window and tests on the **next two months**.
- The real task is *train through October (month 1 of Q4), predict Nov-Dec (months 2-3 of Q4)*. I built folds with that exact shape:
  - **Q2:** train Jan-Apr -> test May-Jun
  - **Q3:** train Jan-Jul -> test Aug-Sep
- Plus two stress-test folds that predict into a quarter the model hasn't seen at all:
  - **X1:** Jan-Jun -> Jul-Aug
  - **X2:** Jan-Aug -> Sep-Oct (most recent data)
- Number of boosting rounds: early stopping on the latest 20% of each training window (time-ordered, never the test months), then refit on the whole window with 15% more rounds.
- Metrics reported on clean test rows and on all test rows (the corrupted labels can't be predicted by any model, they only add noise).

---

## 6. Model iterations

MAPE on clean test rows:

| Model | Q2 | Q3 | X1 | X2 | Mean |
|---|---|---|---|---|---|
| Baseline: rate-card median | 6.04% | 3.47% | 3.62% | 3.48% | 4.15% |
| Log-linear OLS | 2.26% | 3.56% | 2.77% | 3.94% | 3.13% |
| Log-linear OLS + trend | 1.92% | 1.92% | 2.27% | 1.98% | 2.02% |
| LightGBM, no time | 1.84% | 3.01% | 3.03% | 2.95% | 2.71% |
| LightGBM, time as feature | 2.24% | 2.31% | 2.40% | 1.93% | 2.22% |
| LightGBM, detrended | 1.54% | 2.81% | 2.98% | 1.55% | 2.22% |
| LightGBM, detrended, **no label cleaning** | 2.82% | 3.14% | 3.39% | 2.20% | 2.89% |
| LightGBM hybrid | 1.51% | 1.67% | 1.96% | 1.49% | 1.66% |
| hybrid, 31 leaves, no quote_signal | 1.54% | 1.47% | 1.94% | 1.52% | 1.62% |
| + 5 seeds + lane premium (section 6.5) | 1.49% | 1.44% | 1.91% | 1.49% | 1.59% |
| **FINAL: linear trees + 5 seeds + lane premium (section 6.6)** | **1.39%** | **1.35%** | **1.85%** | **1.42%** | **1.50%** |
| Blend: detrended + time-feature | 1.74% | 2.16% | 2.55% | 1.60% | 2.02% |
| Blend: detrended + OLS | 1.51% | 1.97% | 2.34% | 1.57% | 1.85% |

### 6.1 Why plain LightGBM was unstable
- `market_index_day` has a unique value for each day, so the trees can split on it like a **date ID** and memorise each day's price level.
- On a future day, the model borrows the level of a past day with a similar market value, which may have been in a ramp or another quarter.
- Trees also can't extrapolate: with time as a feature they hold the last level flat.

### 6.2 The hybrid (final) model
```
log(rate) = market part (linear)          +  load part (LightGBM)
            market elasticity                 distance, equipment, weight,
            quarter-end ramp x equipment      pickup/delivery coordinates,
            base trend                        circuity, weight flags
```
- The linear part owns everything **time-varying**, so it can extrapolate into Nov-Dec.
- The trees see **no date or market feature**, so they only learn the stable "what does this load cost" structure.
- Easy to explain: *the trees price the load; a transparent market model decides when it costs more.*

### 6.3 Small tuning (mean over Q2, Q3, X2)
| Variant | Mean MAPE |
|---|---|
| default (63 leaves) | 1.560% |
| 31 leaves | 1.540% |
| 127 leaves | 1.584% |
| min_data_in_leaf 100 | 1.602% |
| Huber loss | 1.612% |
| L1 loss | 1.614% |
| drop quote_signal | 1.523% |
| drop weight flags | 1.561% |
| drop circuity / deltas | 1.616% |

- Squared error beats Huber / L1 because the outliers were already removed.
- Geometry features help; weight flags neutral (kept for traceability).

### 6.4 Residual checks on the final approach
- Out-of-fold error by week, equipment and distance quartile: systematic error **under 1%** everywhere.
- Only visible gap: October in fold X2 under-predicted ~1% because the Q4 base step had never been seen. In the real run October is in training, so the Q4 base is known.

---

## 6.5 Bottleneck analysis: where is the remaining error?

Took the out-of-fold log residuals of the model (folds Q2, Q3, X2; residual std 0.0193, MAPE ~1.52%) and split the variance into sources:

| Source | Share of error variance | How measured |
|---|---|---|
| Whole-day level miss | 13.0% | mean residual per date |
| Lane-specific (A->B route) premium | 13.7% | mean residual per lane after removing the day effect, corrected for sampling noise |
| Pickup city alone / delivery city alone | 0.5% / 0.9% | same, per city |
| Equipment / distance / weight segments | bias 0.2-0.3% | mean residual per group |
| Load-level remainder | ~73% | what's left |

**Is the remainder learnable?** No. Its correlation with every feature is below 0.013 (market-index deviation -0.007, quote_signal 0.013, weight 0.006, distance 0.000). It's noise.
-> **Noise floor ~1.3% MAPE.** We're at ~1.5%, so ~85-90% of the learnable signal is already captured.

**Day-level error pattern**
- Sun/Mon under-predicted ~0.5%, Wed/Thu ~0%: rates react less to the weekly market swing than to the slow market level.
- Errors on consecutive days correlate 0.64: a slowly moving level error.

### Lever A: lane premium (adopted)
- Each lane's mean out-of-fold residual in the training window (5-fold inside the window, so the trees haven't already absorbed it), shrunk towards 0 by n / (n + 10).
- Only 0.02-0.04 points better per fold: lane premiums **don't persist strongly** from past months to future months. Most of the 13.7% is noise inside the test window, not a stable lane personality.
- Tried lane x equipment keys: worse (too few loads per key). Shrinkage 5/10/20: 10 best overall.

### Lever B: split market into slow level + weekly swing (rejected)
| Variant | Q2 | Q3 | X1 | X2 | Mean |
|---|---|---|---|---|---|
| current | 1.48% | 1.50% | 1.95% | 1.52% | 1.61% |
| + day-of-week terms | 1.79% | 1.41% | 1.89% | 1.55% | 1.66% |
| trailing 7-day smooth + deviation | 1.58% | 1.43% | 1.62% | 1.58% | 1.55% |
| trailing 14-day smooth + deviation | 1.54% | 1.47% | 1.53% | 1.62% | 1.54% |
| trailing 28-day smooth + deviation | 1.49% | 1.64% | 1.46% | 1.73% | 1.58% |
| centred 7-day smooth + deviation | 1.91% | 1.41% | 1.78% | 1.56% | 1.67% |
- The mean improves only because the hardest fold (X1) improves; the production-like folds (Q2, Q3) do not, and X2 gets worse. It moves error between folds instead of removing it, so it was not adopted.

### Seed averaging (adopted)
- 5 LightGBM models with different random seeds, averaged. Reduces tree variance.

| Version | Q2 | Q3 | X1 | X2 | Mean MAPE | Mean MAE |
|---|---|---|---|---|---|---|
| hybrid (previous final) | 1.538% | 1.473% | 1.937% | 1.524% | 1.618% | $37.47 |
| + 5 seeds | 1.529% | 1.465% | 1.932% | 1.516% | 1.610% | $37.27 |
| + lane premium | 1.503% | 1.451% | 1.921% | 1.502% | 1.595% | $36.90 |
| **+ both (final)** | **1.493%** | **1.442%** | **1.915%** | **1.494%** | **1.586%** | **$36.68** |

Small, but better on every single fold.

### Residual learnability test (the definitive check)
The first decomposition only used simple correlations, which can miss interactions. So I trained a second LightGBM whose only job is to predict the final model's out-of-sample errors (fold Q3 test window, 5-fold CV):

| Features the error model gets | Share of remaining error variance explained |
|---|---|
| all load features | 2.2% |
| all load features except quote_signal | **0.3%** |
| quote_signal + distance | 2.0% |
| quote_signal alone | 0.9% |
| row-level market-index deviation | 0% |

- The only thing left is a weak quote_signal x distance interaction, worth ~0.02 MAPE points at best. Keeping quote_signal in the model made the folds slightly worse (1.52% vs 1.54%), so it's too weak to learn reliably.
- Date features "explain" more (8%) only because they let the error model copy the level of other loads on the same test day, which is future information and not usable.
- First conclusion at this point: "the remaining error is noise". **This turned out to be wrong**, see 6.6.

### 6.6 Other learning algorithms for the load part (a correction)
Tested other learners inside the same hybrid (market part unchanged), same 4 folds:

| Load-part learner | Q2 | Q3 | X1 | X2 | Mean |
|---|---|---|---|---|---|
| gradient-boosted trees (previous) | 1.533% | 1.477% | 1.942% | 1.527% | 1.620% |
| **linear trees** (each leaf fits a linear model) | **1.424%** | **1.391%** | **1.875%** | **1.453%** | **1.536%** |
| extra-trees | 1.731% | 1.642% | 2.066% | 1.700% | 1.785% |
| random forest | 3.92% | 3.75% | 4.06% | 3.69% | 3.86% |
| DART | 8.5% | 7.5% | 6.6% | 8.4% | 7.8% |

- **Why linear trees win:** pricing is smooth (power law in distance, roughly linear in weight). Normal trees approximate a smooth curve with flat steps, so every load gets "rounded" to its step; linear leaves follow the slope inside each region.
- **Lesson:** my ceiling test in 6.5 used normal trees as the error model, so it had the same blind spot. The ceiling estimate was limited by the model class used to measure it.
- Robust to settings: leaves 15/31/63, linear_lambda 0.1/1/10, min_data_in_leaf 40/100 all give 1.53-1.54%. Kept 31 leaves, lambda 1.
- XGBoost / CatBoost not tried: same family as LightGBM's standard trees, and installs were unreliable on this machine; the gain here came from the leaf type, not the library.

Combined with the earlier gains:

| Version | Q2 | Q3 | X1 | X2 | Mean MAPE | Mean MAE |
|---|---|---|---|---|---|---|
| linear trees | 1.423% | 1.385% | 1.867% | 1.446% | 1.530% | $36.32 |
| + 5 seeds | 1.414% | 1.379% | 1.863% | 1.443% | 1.525% | $36.19 |
| + lane premium | 1.394% | 1.361% | 1.849% | 1.423% | 1.507% | $35.78 |
| **+ both (FINAL)** | **1.385%** | **1.352%** | **1.845%** | **1.419%** | **1.500%** | **$35.64** |

**Re-checked the ceiling** with an error model that also uses linear trees (fold Q3, final model test MAPE 1.34%): load features explain only 2.8-3.3% of the remaining error variance, 0.8-1.4% without quote_signal. This time the remaining error really is close to noise.

### 6.7 Final robustness checklist (on the final model's out-of-fold predictions, all 4 folds)

**1. Where the 1.50% is measured.** On test rows with clean labels. On all test rows including the corrupted ones: 3.85%. Every load with rate/mile > $5 (256) or < $1 (253) in the test windows is inside our flagged set (100% overlap); the ratio rule catches 25 more that simple $/mile cut-offs miss.

**High-circuity loads (distance > 1.5x straight line): not an error.** 214 in train, 55 in validation. All very short hauls (median straight-line 52 mi vs 807 overall): with shifted synthetic coordinates, nearby cities get a high ratio. Stated distance is consistent inside each lane (99.8% within +/-13% of the lane median). Distance has a floor at 70.0 mi. Error on circuity 1.5-2.0: 1.47% (normal).

**2. Error by slice (clean labels; overall 1.50%)**
| Slice | MAPE | vs average | Bias |
|---|---|---|---|
| imputed weight (237) | 2.40% | 1.6x | +0.06% |
| imputed market_index (285) | 1.58% | 1.05x | +0.26% |
| negative weight (200) | 1.43% | 0.95x | 0.00% |
| distance <= 100 mi (212) | 1.54% | 1.02x | +0.27% |
| weight < 10k lb (163) | 1.89% | 1.26x | +0.31% |
| weight 44k-47.5k / at 47.5k cap | 1.54% / 1.48% | 1.03x / 0.98x | |
| Dry Van / Flatbed / Reefer | 1.50% / 1.51% / 1.51% | 1.0x | |
| day of week | 1.47-1.53% | 0.98-1.02x | Wed/Thu +0.38% |
| last 0-3 days of a quarter | 1.33-1.47% | ~0.9x | **-0.5% to -0.9%** (under-predicts) |
- No slice is 5-10x worse, so no hidden pricing rule (no minimum-charge floor for short hauls, no overweight tier).
- Imputed weight is worse because weight is genuinely unknown (weight moves price ~+/-3%).

**3. Residual structure trees could miss**
- `load_id` order: IDs are sorted by date; position within a day has no relation to the error (corr -0.007). No leak.
- Lane short-term shocks: a lane's error over the previous 14 days doesn't predict its next error (corr 0.03).
- Day-level errors persist (lag-1 autocorr 0.74): slow level drift, already known.
- quote_signal: its spread shrinks exactly with sqrt(distance) (sd x sqrt(distance) ~ 7.5 in every band), so quote_signal ~ base + noise / sqrt(distance). On short hauls the error correlates 0.27 with |quote_signal - 2.07|, but standardised z = (qs - 2.07) x sqrt(distance) only moves the error by +/-0.3%. Added back to the model:

| Variant | Q2 | Q3 | X1 | X2 | Mean |
|---|---|---|---|---|---|
| final (no quote_signal) | 1.422% | 1.387% | 1.867% | 1.449% | 1.531% |
| + quote_signal | 1.343% | 1.609% | 1.866% | 1.397% | 1.554% |
| + z, abs(z) | 1.451% | 1.914% | 2.099% | 1.383% | 1.712% |
| + all four | 1.481% | 1.893% | 2.080% | 1.389% | 1.711% |

  Helps some folds, badly hurts others: its relationship with price is not stable over time. Rejected.

**4. Loss vs metric.** Picked the best multiplicative correction on the Q folds and checked on the X folds: best factor is **1.000** for MAPE and 1.001 for MAE/RMSE (which makes the X folds slightly worse). Training in log space is already well calibrated.

**5. Holidays.** Day-level error around Memorial Day, July 4 and Labor Day is no different from the neighbouring days (Labor Day's small jump coincides with Sep 1, the start of the Q3 ramp). No evidence for a holiday effect, so no prior is added for Thanksgiving/Christmas.

**Quarter-end tail.** The last 1-3 days of a quarter are under-predicted by 0.5-1.1% (the ramp steepens at the very end). Adding a 4-day tail term: mean 1.531% -> 1.525%, tail bias +0.83% -> +0.58%, but mixed by fold (Q2, X2 slightly worse) and based on ~12 days from three quarters. Not adopted; noted as a known limitation (Dec 28-31 may be ~0.5% low).

### 6.8 Lane premium: random vs time-respecting out-of-fold residuals
Question a reviewer could ask: "why random folds for the lane premium in a time-series problem?" Tested (single-seed linear-tree hybrid, same 4 folds, MAPE on clean labels):

| Lane premium method | Q2 | Q3 | X1 | X2 | Mean |
|---|---|---|---|---|---|
| none | 1.424% | 1.389% | 1.870% | 1.452% | 1.533% |
| **random 5-fold OOF (used)** | 1.395% | 1.352% | 1.849% | 1.414% | **1.502%** |
| contiguous time blocks, 5-fold | 1.370% | 1.351% | 1.887% | 1.432% | 1.510% |
| strictly past-only (walk-forward by month) | 1.350% | 1.455% | 2.079% | 1.395% | 1.570% |
| random + lane x equipment layer | 1.397% | 1.354% | 1.852% | 1.417% | 1.505% |

- The lane premium is meant to capture a *static* trait of a route. Past-only residuals come from models that are forecasting each month, so they also carry that month's level error (not lane-specific) and early months have tiny training sets; it gets worse than no premium on X1.
- Random OOF isolates the lane effect; none of the variants ever uses test labels. It is the only variant that improves every fold. Kept.

### 6.9 Ramp shape check
The quarter-end ramp was assumed linear over 31 days. Tested lengths 21/31/45/60 days x shapes linear / power 1.5 / power 2 (single seed, standard trees):
- 31-day linear: 1.620% mean, best on the production-like folds together with 45-day power-2 (1.616%, within noise).
- 21 and 60 days clearly worse (1.66-1.88%). Kept 31-day linear.

---

## 7. Final model

- `FINAL: hybrid linear-tree x5 seeds + lane premium`
- Trained on all clean Jan-Oct loads: **47,323 rows** (677 removed).
- 1,358 boosting rounds x 5 seeds (linear trees need far fewer rounds); in-sample MAPE 1.06% vs ~1.35-1.85% out-of-fold, a smaller gap than the standard-tree version (0.99% in-sample), so less overfitting.
- Lane premiums for 4,013 lanes, range -2.1% to +1.4% (after shrinkage).
- Market-part coefficients:

| Term | Coefficient | Meaning |
|---|---|---|
| log_market_index | 0.130 | together with the next term: 10% higher index -> ~1.5% higher rate |
| log_market_index_day | 0.023 | small extra weight on the daily mean |
| qe_ramp | 0.023 | Dry Van: +2.3% on the last day of a quarter |
| qe_ramp_flatbed | 0.053 | Flatbed: +7.9% total on the last day |
| qe_ramp_reefer | 0.025 | Reefer: +5.0% total on the last day |
| trend_months | 0.0059 | +0.6% per month base drift |

---

## 8. December chart

- The file only has lane, equipment, weight and date. `FeatureBuilder` fills the rest:
  - coordinates from the city table (Lexington, Fort Wayne are known cities),
  - market index = mean of all validation loads on that date (validation.csv has market readings for every December day; no labels are used).
- Result: **~$831 on Dec 1 -> ~$882 on Dec 31** (unchanged by the load-part upgrades, since the market part drives the date pattern).
  - The weekly zig-zag is the market index's weekly cycle (peaks on Thursdays).
  - The upward drift is the quarter-end/year-end ramp plus the base trend.
- Sanity check against real history on this lane (Dry Van, 17-41k lb): $757-$934, e.g. $864 on Oct 30 at market index 1.01. December predictions sit inside that range.
- Known limitation: the model has no holiday calendar (nothing in the training months to learn it from), so Christmas week is priced like any other week at that market level.

---

## 9. Output checks

- `validation_predictions.csv`: 12,000 rows, `load_id,predicted_rate`, ids match the template, all positive, median $2,075.
- `score.py` passes both files and writes `scorer_results/candidate_december.png`.
- Refactoring the code for readability changed predictions by exactly 0.0.

---

## 10. Things that went wrong along the way (and how they were solved)

| Problem | Fix |
|---|---|
| Slow / dropping internet during pip installs | reused the system's pandas/numpy/scipy (`--system-site-packages`), installed only LightGBM |
| Windows Smart App Control blocks scikit-learn's compiled files | dropped scikit-learn: LightGBM's native `lgb.train` API, metrics and folds written with numpy |
| "6% linear drift" extrapolation failed in one fold (7.8% error) | found market/time confounding, then found the real quarter-end pattern |
| quote_signal looked important (U-shape) | showed it was an artifact of a mis-specified distance curve |
| LightGBM memorised dates through `market_index_day` | hybrid design: trees get no date or market features |

---

## 11. Loom talking points (2-3 min)

1. **Data findings (30 s):** 48k loads Jan-Oct, predict Nov-Dec. Price = distance x equipment x weight x daily market. Biggest discovery: rates ramp ~5% in the last month of every quarter.
2. **Data quality (30 s):** 1.4% corrupted labels with a clean gap, negative weights, capped weights, missing values filled by day, synthetic coordinates, 8 new cities in validation.
3. **Model (40 s):** hybrid. Show why plain LightGBM failed (date memorisation, no extrapolation). Linear market part + trees for load structure. 1.50% MAPE / $35.6 MAE vs 4.15% / $96 for a rate-card baseline. Switching to linear trees was the last big gain; the remaining error is close to noise.
4. **Validation (30 s):** no random split. Expanding-window folds shaped like the real task (train through month 1 of a quarter, predict months 2-3). Unseen-city test.
5. **Code (30 s):** `features.py` (FeatureBuilder handles the December file), `models.py` (GBM hybrid), `experiments.py` (all comparisons), `train.py` / `predict.py`.
