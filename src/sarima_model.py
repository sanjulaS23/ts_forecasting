import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from load_data import load_energy
import warnings

# Ignore warning messages to keep the output clean
warnings.filterwarnings('ignore')


# 1. Load the data and calculate the daily average
series = load_energy()
train_daily = series.resample('D').mean()


# 2. Train the SARIMA model
# order=(2,1,1) defines the non-seasonal ARIMA parameters
# seasonal_order=(1,1,1,7) defines the seasonal parameters
# with a weekly seasonal period of 7 days
def fit_sarima(train):
    model = SARIMAX(
        train,
        order=(2, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model to the training data
    result = model.fit(
        disp=False,
        maxiter=200
    )

    # Display the model summary
    print(result.summary())

    return result


print("Fitting SARIMA model (please wait a moment)...")

sarima_result = fit_sarima(train_daily)


# 3. Forecast the next 30 days
forecast = sarima_result.get_forecast(steps=30)

# Get the predicted values
fc_mean = forecast.predicted_mean

# Get the 95% confidence interval
fc_ci = forecast.conf_int(alpha=0.05)


# 4. Plot the forecasting results
fig, ax = plt.subplots(figsize=(14, 5))

# Plot the last 60 days of historical data
train_daily.iloc[-60:].plot(
    ax=ax,
    label='Historical Data',
    color='#1A4A8A'
)

# Plot the 30-day SARIMA forecast
fc_mean.plot(
    ax=ax,
    label='SARIMA Forecast',
    color='#D85A30'
)

# Display the 95% confidence interval
ax.fill_between(
    fc_ci.index,
    fc_ci.iloc[:, 0],
    fc_ci.iloc[:, 1],
    alpha=0.2,
    color='#D85A30',
    label='95% Confidence Interval'
)

# Add the legend and chart title
ax.legend()
ax.set_title('SARIMA 30-Day Energy Consumption Forecast')

# Adjust the layout
plt.tight_layout()


# Save the forecast plot as an image
plt.savefig(
    'notebooks/sarima_forecast.png',
    dpi=150
)

print("SARIMA forecast plot saved to notebooks/sarima_forecast.png ✓")