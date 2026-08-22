import pandas as pd
import numpy as np


def load_energy(path='data/energy.csv'):
    # Read the CSV file using Pandas
    df = pd.read_csv(path)

    # Convert the 'Datetime' column to the correct datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Remove duplicate timestamps and keep only the first occurrence
    df = df.drop_duplicates(subset=['Datetime'])

    # Set 'Datetime' as the index and sort the data chronologically
    df = df.set_index('Datetime').sort_index()

    # Select only the electricity consumption data for the AEP region
    series = df['AEP_MW'].copy()

    # Check for missing timestamps
    full_range = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq='h'
    )

    missing = full_range.difference(series.index)
    print(f"Missing timestamps: {len(missing)}")

    # Fill missing values using forward fill
    series = series.reindex(full_range).ffill()

    return series


# Run the function and check the loaded data
if __name__ == "__main__":
    series = load_energy()

    print("Dataset Loaded Successfully!")
    print(series.describe())