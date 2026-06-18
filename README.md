# Predicting-F1-Pit-Stops
We worked with a synthetically generated, telemetry‑style F1 dataset to predict whether a car would pit on the following lap.

# Dependencies

To view the report without running code, download `analysis.html`.

To run the notebook, install:
1. Matplotlib.
2. Numpy.
3. Pandas.
4. `shap`.
5. `sklearn` (scikit learn).
6. `xgboost`.

# Issues with the data

Several domain‑violating inconsistencies show the dataset is synthetic:
1. Double pitting occurs in $\approx 25%$ of pit laps (physically unrealistic).
2. Base pit rates are far too high ($\approx 14%$ pit laps, $\approx 20%$ laps preceeding a pit stop).
3. `Stint` numbers decrease in a non-trivial portion of laps (impossible).
4. The features `LapTime_Delta` and `Position_Change` do not match their definitions.
5. Driver counts per race range from 4 to 856 (real F1 has 20).
6. Sample size is orders of magnitude above what is physically possible.
7. Missing key features (weather, safety car, gaps to other cars).

# Linear modelling

We trained a baseline linear model (logisitic regression) using two feature engineering strategies:
1. One-hot encoding: high-dimensional feature space (928 features); $\approx 60%$ recall, $\approx 60%$ precision, and $\approx 85%$ accuracy.
2. Feature aggregation: similar recall ($\approx 60%$), but lower precision ($\approx 44%$); $\approx 77%$ accuracy.

Due to class imbalance, accuracy is not a particularly insightful metric in isolation.
We analysed the learned weightings for the model, and gave some possible interpretations.

# Tree modelling

We also trained a tree-based ensemble (XGBoost). This gave better performance:
- $\approx 72%$ recall,
- $\approx 75%$ precision,
- $\approx 90%$ accuracy.

Shap analysis showed:

1. Global feature importance differed from conditional importance (e.g., `PitStop` becomes influential when `PitStop==1`).
2. *Beeswarm* plots showed some similar trends to the linear model; but also non-linear interactions not capured by logistic regression.

# Tree error analysis

A simple first-order error analysis using **lift** values showed that the model failures were not uniformly distributed across the test set. Lift values consistently fell outisde $99%$ empirical null intervals, indicating concentrated error regions.
