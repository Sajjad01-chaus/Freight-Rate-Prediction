# Model comparison (time-based folds)

## Test rows: clean

| Model | Mean MAE | Mean MAPE | Mean RMSE | Holdout F3 MAE | Holdout F3 MAPE | F3 Bias |
|---|---|---|---|---|---|---|
| Baseline: rate-card median | 96.00 | 4.15% | 130.92 | 77.15 | 3.48% | -0.72% |
| Log-linear OLS | 72.99 | 3.13% | 98.18 | 91.32 | 3.94% | -3.80% |
| Log-linear OLS + trend | 46.45 | 2.02% | 66.41 | 44.53 | 1.98% | -0.53% |
| LightGBM (trend=none) | 61.16 | 2.71% | 88.55 | 71.72 | 2.95% | -2.73% |
| LightGBM (trend=feature) | 50.62 | 2.22% | 75.97 | 45.71 | 1.93% | -1.33% |
| LightGBM (trend=detrend) | 52.10 | 2.22% | 77.92 | 34.61 | 1.55% | +0.13% |
| LightGBM (trend=detrend) [no label cleaning] | 65.38 | 2.89% | 94.89 | 49.47 | 2.20% | +0.29% |
| LightGBM (trend=hybrid) | 38.12 | 1.66% | 55.64 | 33.43 | 1.49% | -0.52% |
| Hybrid, linear trees (single model) | 36.32 | 1.53% | 52.29 | 33.46 | 1.45% | -0.57% |
| FINAL: hybrid linear-tree x5 seeds + lane premium | 35.64 | 1.50% | 51.37 | 32.79 | 1.42% | -0.57% |
| Blend: LightGBM (trend=detrend) + LightGBM (trend=feature) | 46.88 | 2.02% | 72.22 | 36.67 | 1.60% | -0.60% |
| Blend: LightGBM (trend=detrend) + Log-linear OLS + trend | 43.60 | 1.85% | 64.32 | 36.07 | 1.57% | -0.21% |

## Test rows: all

| Model | Mean MAE | Mean MAPE | Mean RMSE | Holdout F3 MAE | Holdout F3 MAPE | F3 Bias |
|---|---|---|---|---|---|---|
| Baseline: rate-card median | 148.54 | 6.43% | 638.51 | 132.86 | 5.99% | +0.77% |
| Log-linear OLS | 125.76 | 5.39% | 632.41 | 146.70 | 6.35% | -2.37% |
| Log-linear OLS + trend | 99.54 | 4.36% | 623.55 | 100.55 | 4.50% | +0.95% |
| LightGBM (trend=none) | 113.99 | 4.97% | 628.50 | 127.34 | 5.39% | -1.29% |
| LightGBM (trend=feature) | 103.59 | 4.52% | 625.04 | 101.71 | 4.43% | +0.14% |
| LightGBM (trend=detrend) | 105.03 | 4.56% | 623.33 | 90.74 | 4.09% | +1.61% |
| LightGBM (trend=detrend) [no label cleaning] | 118.21 | 5.23% | 626.28 | 105.36 | 4.73% | +1.78% |
| LightGBM (trend=hybrid) | 91.29 | 4.00% | 621.72 | 89.56 | 4.02% | +0.95% |
| Hybrid, linear trees (single model) | 89.53 | 3.87% | 622.02 | 89.59 | 3.97% | +0.91% |
| FINAL: hybrid linear-tree x5 seeds + lane premium | 88.86 | 3.84% | 621.90 | 88.92 | 3.94% | +0.91% |
| Blend: LightGBM (trend=detrend) + LightGBM (trend=feature) | 99.90 | 4.34% | 623.53 | 92.79 | 4.12% | +0.87% |
| Blend: LightGBM (trend=detrend) + Log-linear OLS + trend | 96.70 | 4.19% | 622.43 | 92.20 | 4.10% | +1.27% |
