import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from load_data import load_energy


# 1. Load the data and prepare it for scaling
series = load_energy()

# Calculate the daily average energy consumption
train_daily = series.resample('D').mean()

# Scale the data to a range between 0 and 1
# LSTM models generally perform better when the input values
# are normalized to a smaller range
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    train_daily.values.reshape(-1, 1)
).flatten()


# 2. Create sequences for the LSTM model
# Use the previous 30 days of data to predict the next 7 days
SEQ_LEN = 30       # Number of previous days used as input
PRED_STEPS = 7     # Number of future days to predict


def create_sequences(data, seq_len, pred_steps):
    X, y = [], []

    for i in range(len(data) - seq_len - pred_steps):
        # Input: previous 30 days
        X.append(data[i:i + seq_len])

        # Target: following 7 days
        y.append(data[i + seq_len:i + seq_len + pred_steps])

    # Add an extra dimension for the LSTM input feature
    return np.array(X)[..., np.newaxis], np.array(y)


# Create input sequences and target values
X, y = create_sequences(
    scaled_data,
    SEQ_LEN,
    PRED_STEPS
)


# Split the data into training and testing sets
# 80% is used for training and 20% for testing
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# Convert the training data into PyTorch tensors
Xtr = torch.tensor(
    X_train,
    dtype=torch.float32
)

ytr = torch.tensor(
    y_train,
    dtype=torch.float32
)


# Create a DataLoader for batch training
loader = DataLoader(
    TensorDataset(Xtr, ytr),
    batch_size=64,
    shuffle=True
)


# 3. Define the LSTM forecasting model
class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        pred_steps=7
    ):
        super().__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Fully connected layers used to generate predictions
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, pred_steps)
        )

    def forward(self, x):
        # Pass the input sequence through the LSTM
        out, _ = self.lstm(x)

        # Use the final hidden state to make the prediction
        out = self.fc(out[:, -1, :])

        return out


# Create the LSTM model
model = LSTMForecaster()

# Define the loss function
criterion = nn.MSELoss()

# Define the Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# 4. Train the LSTM model
print("Training LSTM model (this may take a minute)...")

epochs = 20

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for xb, yb in loader:

        # Clear the previous gradients
        optimizer.zero_grad()

        # Generate predictions
        pred = model(xb)

        # Calculate the loss
        loss = criterion(pred, yb)

        # Calculate gradients
        loss.backward()

        # Update the model parameters
        optimizer.step()

        total_loss += loss.item()

    # Display the loss every 5 epochs
    if (epoch + 1) % 5 == 0:
        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.5f}"
        )


print("LSTM Model Training Completed Successfully! ✓")