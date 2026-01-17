import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(csv_path):
    # 1. Load Data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. Filter (Optional - purely to be safe)
    # Ensure we only use data within the intended linear region
    original_len = len(df)
    df = df[(df['dac_input'] >= 140) & (df['dac_input'] <= 170)]
    print(f"Filtered data from {original_len} to {len(df)} samples (Range 140-170).")

    # 3. Normalization (CRITICAL STEP)
    # Neural Networks work best with data in range [0, 1] or [-1, 1]

    # Input: Map 140-170 to 0-1
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    X = df['dac_input'].values.reshape(-1, 1)
    X_scaled = input_scaler.fit_transform(X)

    # Output: Map Sensor values (e.g., 2000-5000) to 0-1
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    y = df['sensor_output'].values.reshape(-1, 1)
    y_scaled = output_scaler.fit_transform(y)

    return X_scaled, y_scaled, input_scaler, output_scaler

class VLCDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors (Float32)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    csv_file = 'final_merged_dataset.csv'
    
    # Check if file exists
    import os
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Load and Preprocess
    X_scaled, y_scaled, input_scaler, output_scaler = load_and_process_data(csv_file)

    # 4. Train/Test Split
    # Split 80% for training, 20% for validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Create PyTorch Datasets
    train_dataset = VLCDataset(X_train, y_train)
    test_dataset = VLCDataset(X_test, y_test)

    # Create DataLoaders (Batch size = 32 or 64)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("DataLoaders created successfully.")
    
    # Verify one batch
    inputs, targets = next(iter(train_loader))
    print(f"Sample Batch - Input shape: {inputs.shape}, Target shape: {targets.shape}")

    # Inference Example
    print("\n--- Inference Example ---")
    # Example: Model predicts signal 0.8
    predicted_norm = np.array([[0.8]]) 
    
    # Convert back to real value (140-170)
    real_dac_val = input_scaler.inverse_transform(predicted_norm)
    print(f"Predicted Normalized: 0.8 -> Real DAC Value to send: {int(real_dac_val)}")

if __name__ == "__main__":
    main()
