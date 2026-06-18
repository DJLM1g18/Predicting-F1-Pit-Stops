# Predicting-F1-Pit-Stops
We worked with a synthetically generated, telemetry‑style F1 dataset to predict whether a car would pit on the following lap.

# Dependencies

If you wish to view this report without downloading the notebook and running code, you can download `analysis.html`, which is an export of the notebook file.

If you wish to run the notebook, you'll need the following packages:
1. Matplotlib.
2. Numpy.
3. Pandas.
4. `shap`.
5. `sklearn` (scikit learn).

# Issues with the data

In this section, we noted several oddities about the data. This included:;
1. Double pitting events were unusually high ($\approx 25\%$ of the data). F1 drivers rarely pit two laps in a row.
2. High pitting base rates: $\approx 14\%$ of the samples provided were pit laps; $\approx 20\%$ of the samples provided preceeded a pit stop. This is much higher than rates of pitting for actual F1 drivers.
3. `Stint` should increase monotonically with `LapNumber` for a given race. This isn't true over a non-negligable portion of the data.
4. The features `LapTime_Delta` and `Position_Change` do not behave as expected; we'd expect `LapTime_Delta` to be the simple arithmetic difference of two consecutive lap times. Similarly, we'd expect `Position_Change` to be the simple arithmetic different of consecutive `Position` values. This doesn't seem to be the case.
5. Massive driver numbers per race: one race has only 4 drivers, while another has 856. This is far outside the usual 20 driver count.
6. The number of samples are orders of magnitude higher than what is physically possible.
7. Missing features: features that would be very important for making pit stop decisions, such as those related to the weather, or safety car information, are not included.

Overall, it must be the case that the data is synthetic.

