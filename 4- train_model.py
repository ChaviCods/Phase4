import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import mlflow
import mlflow.pytorch

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 4      # Number of features in the input data (excluding 'Close')
hidden_size = 50     # Number of hidden units in the LSTM
num_layers = 2      # Number of LSTM layers
output_size = 1      # Number of output units (predicting 'Close' price)
num_epochs = 50     # You might want to adjust this based on previous discussions
batch_size = 64
learning_rate = 0.001
sequence_length = 20  # Length of the input sequences

# Load data from CSV
data = pd.read_csv("./cleaned_VALE3_data.csv", index_col='Date')

# Create sequences and labels
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(data.iloc[i:(i + sequence_length), 1:].values)  # Features (excluding 'Close')
        y.append(data.iloc[i + sequence_length, 0])  # 'Close' price
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(data, sequence_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
train_X, train_y = X[:train_size], y[:train_size]
test_X, test_y = X[train_size:], y[train_size:]

# Convert to PyTorch tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).view(-1, 1)  
test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32).view(-1, 1)  

# Create DataLoaders
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# LSTM model 
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm1(x, (h0, c0))
        out = self.fc1(out[:, -1, :])

        out, _ = self.lstm2(x, (h0, c0))
        out = self.fc2(out[:, -1, :])
        return out

# Evaluate the model
def evaluate_model(model, criterion):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Store predictions and labels for other metrics
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate metrics (using unscaled values)
    mae = mean_absolute_error(np.array(all_labels), np.array(all_predictions))  # Convert lists to arrays
    rmse = root_mean_squared_error(np.array(all_labels), np.array(all_predictions))  # Convert lists to arrays
    mape = np.mean(np.abs((np.array(all_labels) - np.array(all_predictions)) / np.array(all_labels))) * 100
    directional_accuracy = np.mean((np.array(all_labels[1:]) - np.array(all_labels[:-1])) * (np.array(all_predictions[1:]) - np.array(all_predictions[:-1])) > 0) * 100

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"Directional Accuracy: {directional_accuracy:.4f}%")

    mlflow.log_metric("test_loss", average_test_loss)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("directional_accuracy", directional_accuracy)

    return average_test_loss

# Training the model
def train_model():
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("LSTM Stock Price Prediction")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("intermediate_networks", [
            nn.LSTM.__name__, 
            nn.Linear.__name__
        ])
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for i, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)

            # Evaluate the model at the end of each epoch
            test_loss = evaluate_model(model, criterion)
            mlflow.log_metric("test_loss", test_loss, step=epoch)

        mlflow.pytorch.log_model(model, "lstm_stock_price_model")
        torch.save(model.state_dict(), "lstm_stock_price_model.pth")

# Train the model 
train_model()