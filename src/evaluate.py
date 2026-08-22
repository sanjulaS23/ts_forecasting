import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_energy


# 1. Load the energy consumption data
series = load_energy()

# Calculate the daily average energy consumption
train_daily = series.resample('D').mean()


# 2. Split the last 30 days as the test data
# The remaining data will be used for training
train = train_daily[:-30]
test = train_daily[-30:]

print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")


# 3. Create a plot to compare the training and test data
plt.figure(figsize=(14, 6))

# Plot the last 90 days of training data
train.iloc[-90:].plot(
    label='Historical Train Data',
    color='#1A4A8A'
)

# Plot the actual test data
test.plot(
    label='Actual Test Data',
    color='black',
    linewidth=2
)

plt.title('Energy Consumption: Train vs Test Split')
plt.xlabel('Date')
plt.ylabel('AEP_MW')
plt.legend()

# Adjust the layout
plt.tight_layout()


# Save the comparison plot as an image
plt.savefig(
    'notebooks/model_comparison.png',
    dpi=150
)

print("Model comparison plot saved to notebooks/model_comparison.png ✓")