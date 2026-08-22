import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from load_data import load_energy


# 1. Load the energy consumption data
# Use the load_energy function that we created earlier
series = load_energy()

# Convert the hourly data into daily average values
# This makes the data easier to analyze
train_daily = series.resample('D').mean()


# 2. Seasonal Decomposition
# Separate the time series into:
# - Trend
# - Seasonality
# - Residuals
#
# period=7 is used to identify a weekly seasonal pattern
decomp = seasonal_decompose(
    train_daily,
    model='additive',
    period=7
)

# Create four subplots for the original series,
# trend, seasonality, and residuals
fig, axes = plt.subplots(
    4, 1,
    figsize=(14, 10),
    sharex=True
)

train_daily.plot(
    ax=axes[0],
    title='Original Series',
    color='#1A4A8A'
)

decomp.trend.plot(
    ax=axes[1],
    title='Trend',
    color='#D85A30'
)

decomp.seasonal.plot(
    ax=axes[2],
    title='Seasonality',
    color='#1D9E75'
)

decomp.resid.plot(
    ax=axes[3],
    title='Residuals',
    color='#7F77DD'
)

# Remove the x-axis labels from each subplot
for ax in axes:
    ax.set_xlabel('')

# Adjust the spacing between the plots
plt.tight_layout()

# Save the decomposition plot as an image
plt.savefig(
    'notebooks/decomposition.png',
    dpi=150
)

print("Decomposition plot saved to notebooks/decomposition.png ✓")


# 3. Stationarity Test using the Augmented Dickey-Fuller (ADF) Test
def adf_test(series, name=''):
    # Remove missing values before performing the ADF test
    result = adfuller(
        series.dropna(),
        autolag='AIC'
    )

    print(f"\n--- ADF Test: {name} ---")
    print(f"  Statistic : {result[0]:.4f}")
    print(f"  p-value   : {result[1]:.4f}")

    # If the p-value is less than or equal to 0.05,
    # the series is considered stationary
    verdict = (
        "Stationary ✓"
        if result[1] <= 0.05
        else "Non-stationary — apply differencing"
    )

    print(f"  Result    : {verdict}")


# Perform the ADF test on the original daily series
adf_test(
    train_daily,
    'Raw Daily Series'
)

# Perform the ADF test after first-order differencing
adf_test(
    train_daily.diff().dropna(),
    'After 1st Differencing'
)