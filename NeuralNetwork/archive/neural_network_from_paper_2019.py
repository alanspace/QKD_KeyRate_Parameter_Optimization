# %%
# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import json
from tqdm import tqdm
import torch.nn.functional as F  # Add this import at the top

import os
import sys
# Get the notebook's directory
notebook_dir = os.getcwd()
# Add parent directory to path
project_root = os.path.dirname(notebook_dir)
sys.path.append(project_root)

# %% [markdown]
#  checking if the MPS (Metal Performance Shaders) backend is available

# %%
# Check if MPS is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS device not available. Check if PyTorch and macOS set up correctly.")

# Set the device to MPS
device = torch.device("mps")  # Use GPU on M2 Pro

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from QKD_Functions.QKD_Functions import (
    calculate_factorial,
    calculate_tau_n,
    calculate_eta_ch,
    calculate_eta_sys,
    calculate_D_mu_k,
    calculate_n_X_total,
    calculate_N,
    calculate_n_Z_total,
    calculate_e_mu_k,
    calculate_e_obs,
    calculate_h,
    calculate_lambda_EC,
    calculate_sqrt_term,
    calculate_tau_n,
    calculate_n_pm, 
    calculate_S_0,
    calculate_S_1,
    calculate_m_mu_k,
    calculate_m_pm,
    calculate_v_1,
    calculate_gamma,
    calculate_Phi,
    calculate_LastTwoTerm,
    calculate_l,
    calculate_R,
    experimental_parameters,
    other_parameters,
    calculate_key_rates_and_metrics,
    penalty, 
    objective,
    objective_with_logging
)

# %%
class BB84NN(nn.Module):
    def __init__(self):
        super(BB84NN, self).__init__()
        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256, 512)  # Change 64 -> 32 to match checkpoint
        self.fc4 = nn.Linear(512, 1024)  # Change 64 -> 16 to match checkpoint
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)  # Change 32 -> 16 to match checkpoint
        self.fc9 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        
        return x

# %%
# Load dataset
with open('../Training_Data/qkd_grouped_dataset_20250213_110036.json', 'r') as f:
    data_by_nx = json.load(f)

print(f"The overall dataset contains {len(data_by_nx)} entries.")

# Verify the length of the list associated with the first key
first_key = list(data_by_nx.keys())[0]
print(f"The number of entries associated with the first key ({first_key}) is: {len(data_by_nx[first_key])}")

# Flatten the data structure and filter
cleaned_data = []
for n_x, entries in data_by_nx.items():
    cleaned_data.extend([
        item for item in entries
# if item["key_rate"] > 0 and 
if item["e_1"] * 100 <= 200  # Only positive key rates and fiber lengths <= 200 km
    ])

# Optional: Verify the cleaned dataset
if not cleaned_data:
    print("No valid data after filtering.")
else:
    print(f"Filtered dataset contains {len(cleaned_data)} entries.")
    print("\nSample entry from the cleaned dataset:")
    print(json.dumps(cleaned_data[0], indent=2))
    print("\nNumber of unique n_X values:", len(data_by_nx))

# %%
Y = []
for item in cleaned_data:
    mu_1, mu_2, P_mu_1, P_mu_2, P_X = item['optimized_params'].values()
    # Constraints applied but not stored in Y
    # P_mu_3 = max(1 - (P_mu_1 + P_mu_2), 1e-6)
    # P_Z = max(1 - P_X, 1e-6)
    
    # Store only the first five parameters
    Y.append([mu_1, mu_2, P_mu_1, P_mu_2, P_X])

Y = np.array(Y, dtype=np.float32)

# %%
# Separate features (X) and targets (Y)
X = np.array([[item['e_1'], item['e_2'], item['e_3'], item['e_4']] for item in cleaned_data], dtype=np.float32)
Y = np.array([list(item['optimized_params'].values()) for item in cleaned_data ], dtype=np.float32)  # Flatten dictionary

# Shuffle the data
from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=42)

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform on training data
X_val = scaler.transform(X_val)  # Transform validation data using the same scaler

from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler()  # Scale targets to [0, 1]
Y_train = y_scaler.fit_transform(Y_train)
Y_val = y_scaler.transform(Y_val)

import joblib
joblib.dump(scaler, 'scaler.pkl')  # Save StandardScaler
joblib.dump(y_scaler, 'y_scaler.pkl')  # Save MinMaxScaler

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")

# %%
# Convert dataset to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Initialize model
model = BB84NN().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer

# Learning rate scheduler
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

# %%
def validate_and_check_constraints(model, val_loader, criterion, device, bounds):
    model.eval()
    total_loss = 0
    constraint_violations = 0

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            predictions = model(X_batch)
            
            # Calculate loss
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item()
            
            # Check constraints
            for i, bound in enumerate(bounds):
                lower, upper = bound
                # Check if any predictions are out of the specified bounds
                if (predictions[:, i] < lower).any() or (predictions[:, i] > upper).any():
                    constraint_violations += 1

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    if constraint_violations == 0:
        print("All constraints satisfied.")
    else:
        print(f"Constraints violated in {constraint_violations} batches.")

    return avg_loss, constraint_violations

# %%
from tqdm import tqdm

num_epochs = 5000
best_loss = float('inf')  # Initialize best loss to infinity
patience = 20 # Number of epochs to wait before stopping
best_val_loss = float('inf')
early_stop_counter = 0
early_stopping_patience = 100  # Early stopping patience

# Initialize placeholders for final outputs
final_train_loss = None
final_val_loss = None

# Bounds for predicted parameters (adjust as needed)
bounds = [
    (4e-4, 0.9),  
    (2e-4, 0.5),  
    (1e-12, 1.0 - 1e-12),  
    (1e-12, 1.0- 1e-12),  
    (1e-12, 1.0- 1e-12),  
]

# %%
# Fixed QKD Parameters (create this OUTSIDE the loss function)
qkd_params = (0.2, 0.1, 6e-7, 1e-10, 1e-15, 1.16, 0.01, 1) #alpha, eta_Bob, P_dc_value, epsilon_sec, 

# Fixed parameters
alpha = 0.2
eta_Bob = 0.1
P_dc_value = 6e-7
epsilon_sec = 1e-10
epsilon_cor = 1e-15
f_EC = 1.16
e_mis = 5e-3
P_ap = 4e-2
n_event = 1

parameter_criterion = nn.MSELoss()

def compute_loss(predictions, targets, X_batch, n_X):
    # Calculate standard loss (e.g., MSE)
    # loss = criterion(predictions, targets)
    
    parameter_loss = nn.MSELoss()(predictions, targets)
    
    return parameter_loss

# %%


def validate(model, val_loader, criterion, device):
    """
    Perform validation and calculate average loss.
    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computation.
    Returns:
        float: Average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)  # Average validation loss
    return val_loss


# %%
def check_constraints(predictions, bounds):
    # """Checks if predictions satisfy the given bounds."""
    # for i, (lower, upper) in enumerate(bounds):
    #     if not (torch.all(predictions[:, i] >= lower) and torch.all(predictions[:, i] <= upper)):
    #         return False
    return True

max_retries = 1

for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    model.train()
    train_loss = 0  # Track training loss for this epoch

    # Inside the training
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        print("Shape of X_batch:", X_batch.shape)
        print("Shape of Y_batch:", Y_batch.shape)
        # Print the first few values to check their range
        print("Sample from X_batch:", X_batch[0])
        print("Sample from Y_batch:", Y_batch[0])
        optimizer.zero_grad()
        
        predictions = model(X_batch.to(device))
        # batch_loss = compute_loss(predictions, Y_batch.to(device), bounds=bounds)
        batch_loss = compute_loss(predictions, Y_batch.to(device), X_batch, 1e8)
        batch_loss.backward()
        
        # Gradient clipping to avoid exploding gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += batch_loss.item()
        
    train_loss /= len(train_loader)

    # Validation step
    current_val_loss = validate(model, val_loader, criterion, device)
    # val_loss, violations = validate_and_check_constraints(model, val_loader, criterion, device, bounds)
    # Print training and validation loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {current_val_loss:.4f}")

    # # Learning rate adjustment
    # scheduler.step(current_val_loss)
        
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        early_stop_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stop_counter += 1
        
    if early_stop_counter >= patience:
        print("Early stopping triggered. Training complete.")
        break

# %%
current_val_loss      

# %%
# Load the trained model for inference
model = BB84NN().to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("Model loaded successfully!")

# %% [markdown]
# 5. Evaluation and Plotting

# %%
# # Load the trained model
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define fiber length range and fixed n_X value
fiber_lengths = np.linspace(0.1, 200, 200)  # Fiber lengths from 0.1 km to 200 km
P_dc_value = 6e-7  # Example value (adjust as needed)
e_mis = 5e-3       # Example misalignment error (adjust as needed)
target_nx = 1e8   # Fixed n_X value

# Prepare inputs for the neural network
predicted_params_list = []

for L in fiber_lengths:
    e_1 = L / 100  # Normalized fiber length
    e_2 = -np.log10(P_dc_value)  # Dark count processing
    e_3 = e_mis * 100  # Misalignment error
    e_4 = np.log10(target_nx)  # Log-scaled detected events
    
    # Construct input tensor
    X = torch.tensor([[e_1, e_2, e_3, e_4]], dtype=torch.float32).to(device)
    
    # Perform prediction
    with torch.no_grad():
        params = model(X).cpu().numpy()[0]
        predicted_params_list.append(params)

# Convert predicted parameters to numpy array
predicted_params_array = np.array(predicted_params_list)

# Display example predictions
print("Example Predicted Parameters:")
for i in range(100):
    print(f"Fiber Length {fiber_lengths[i]:.1f} km -> {predicted_params_array[i]}")

# %%
for i in range(len(predicted_params_list)):
    mu_1, mu_2, P_mu_1, P_mu_2, P_X = predicted_params_list[i]

    # predicted_params_list[i] = [mu_1, mu_2, P_mu_1, P_mu_2, P_X, P_mu_3, P_Z]
    predicted_params_list[i] = [mu_1, mu_2, P_mu_1, P_mu_2, P_X]

# %%
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from QKD_Functions.QKD_Functions import objective

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = BB84NN().to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Load the dataset
with open("../Training_Data/qkd_grouped_dataset_20250213_110036.json", 'r') as f:
    dataset = json.load(f)

# Select an n_X value
target_nx = 1e8
nx_key = str(float(target_nx))
if nx_key not in dataset:
    raise ValueError(f"No data found for n_X = {target_nx}")

optimized_data = dataset[nx_key]

# Extract fiber lengths, optimized key rates, and parameters
fiber_lengths = np.array([entry["fiber_length"] for entry in optimized_data])
optimized_key_rates = np.array([entry["key_rate"] for entry in optimized_data])
optimized_params_array = np.array([list(entry["optimized_params"].values()) for entry in optimized_data])

# Predict parameters and key rates
predicted_params_list = []
predicted_key_rates = []
for L in fiber_lengths:
    e_1 = L / 100
    e_2 = -np.log10(6e-7)
    e_3 = 5e-3 * 100
    e_4 = np.log10(target_nx)
    X = torch.tensor([[e_1, e_2, e_3, e_4]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        params = model(X).cpu().numpy()[0]
        predicted_params_list.append(params)
        key_rate = objective(params, L, target_nx, alpha=0.2, eta_Bob=0.1, P_dc_value=6e-7, epsilon_sec=1e-10, epsilon_cor=1e-15, f_EC=1.16, e_mis=5e-3, P_ap=4e-2, n_event=1)[0]
        predicted_key_rates.append(key_rate)

predicted_params_array = np.array(predicted_params_list)
predicted_key_rates = np.array(predicted_key_rates)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot key rates comparison on the left
ax1.plot(fiber_lengths, np.log10(optimized_key_rates), 'b-', label="Optimized Key Rate")
ax1.plot(fiber_lengths, np.log10(predicted_key_rates), 'r--', label="Predicted Key Rate (NN)")
ax1.set_title('Comparison of Key Rates')
ax1.set_xlabel('Fiber Length (km)')
ax1.set_ylabel('log10(Key Rate)')
ax1.legend()
ax1.grid(True)

# Plot parameters comparison on the right
labels = ['mu_1', 'mu_2', 'P_mu_1', 'P_mu_2', 'P_X']
colors = ['blue', 'green', 'red', 'purple', 'orange']
for i in range(5):
    ax2.plot(fiber_lengths, optimized_params_array[:, i], label=f'Optimized {labels[i]}', color=colors[i], linestyle='-')
    ax2.plot(fiber_lengths, predicted_params_array[:, i], label=f'Predicted {labels[i]}', color=colors[i], linestyle='--')

ax2.set_title('Comparison of Parameters')
ax2.set_xlabel('Fiber Length (km)')
ax2.set_ylabel('Parameter Values')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


