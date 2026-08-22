import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from load_data import load_energy


# 1. Load the data and calculate the daily average
series = load_energy()
df = series.resample('D').mean().reset_index()

# Rename the columns for Prophet
# 'ds' represents the date/time
# 'y' represents the value to be predicted
df.columns = ['ds', 'y']


# 2. Create and train the Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Train the model using the prepared data
model.fit(df)


# 3. Forecast the next 30 days
# Create a dataframe containing the future dates
future = model.make_future_dataframe(periods=30)

# Generate predictions for the future dates
forecast = model.predict(future)


# 4. Plot the forecast results
fig = model.plot(forecast)

plt.title('Prophet 30-Day Energy Consumption Forecast')
plt.tight_layout()

# Save the forecast plot as an image
plt.savefig(
    'notebooks/prophet_forecast.png',
    dpi=150
)

print("Prophet forecast plot saved to notebooks/prophet_forecast.png ✓")


# 5. Plot the trend and seasonal components separately
fig2 = model.plot_components(forecast)

# Save the components plot as an image
plt.savefig(
    'notebooks/prophet_components.png',
    dpi=150
)

print("Prophet components plot saved to notebooks/prophet_components.png ✓")