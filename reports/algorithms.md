# Load-part algorithm comparison (same hybrid design, single seed, no lane premium)

| Algorithm | Q2 | Q3 | X1 | X2 | Mean MAPE | Mean MAE |
|---|---|---|---|---|---|---|
| LightGBM, linear trees | 1.42% | 1.39% | 1.87% | 1.45% | 1.53% | $36.32 |
| LightGBM, standard trees | 1.54% | 1.47% | 1.94% | 1.52% | 1.62% | $37.47 |
| XGBoost, 31 leaves | 1.53% | 1.47% | 1.92% | 1.52% | 1.61% | $37.34 |
| XGBoost, depth 6 | 1.55% | 1.48% | 1.94% | 1.53% | 1.63% | $37.60 |
| Extra-trees (LightGBM) | 1.73% | 1.64% | 2.07% | 1.70% | 1.78% | $40.23 |
| Random forest (LightGBM) | 3.92% | 3.74% | 4.06% | 3.69% | 3.86% | $86.23 |
