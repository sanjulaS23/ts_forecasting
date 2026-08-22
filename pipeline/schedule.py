import pandas as pd
from datetime import datetime
from prefect import flow, task
from prophet import Prophet


@task(retries=2, retry_delay_seconds=30)
def fetch_latest_data():
    # Read the latest energy consumption data from the CSV file
    # In a production environment, this could be fetched from a database or API
    series = pd.read_csv(
        'data/energy.csv',
        index_col=0,
        parse_dates=True
    )['AEP_MW']

    print(f"Fetched {len(series)} rows successfully ✓")

    return series


@task
def validate_data(series):
    # Check the data for missing values or other quality issues
    if series.isnull().mean() > 0.05:
        raise ValueError(
            "Data quality issue: >5% missing values detected!"
        )

    print("Data Validation passed successfully ✓")

    return series


@task
def retrain_prophet(series):
    # Retrain the Prophet model using the latest energy consumption data
    df = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })

    model = Prophet(
        changepoint_prior_scale=0.05
    )

    model.fit(df)

    print("Prophet model retrained successfully ✓")

    return model


@task
def generate_forecast(model, days=30):
    # Generate a forecast for the next 30 days
    future = model.make_future_dataframe(
        periods=days,
        freq='D'
    )

    forecast = model.predict(future)

    # Select the required forecast columns
    out = forecast[
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].tail(days)

    # Create the forecasts directory if it does not exist
    import os
    os.makedirs(
        'forecasts',
        exist_ok=True
    )

    # Create a filename using the current date
    path = f'forecasts/forecast_{datetime.now().date()}.csv'

    # Save the forecast results as a CSV file
    out.to_csv(
        path,
        index=False
    )

    print(
        f"Forecast successfully saved to -> {path} ✓"
    )

    return out


@flow(name="daily-energy-forecast-flow")
def daily_pipeline():
    print("--- Starting Daily Energy Forecast Pipeline ---")

    # Step 1: Fetch the latest data
    raw_data = fetch_latest_data()

    # Step 2: Validate the data
    clean_data = validate_data(raw_data)

    # Step 3: Retrain the Prophet model
    model = retrain_prophet(clean_data)

    # Step 4: Generate the 30-day forecast
    forecast_results = generate_forecast(model)

    print("--- Pipeline Completed Successfully! ---")


if __name__ == "__main__":
    # Run the pipeline locally once for testing
    daily_pipeline()