# %% [markdown]
# PyTorch with Metal Performance Shaders (MPS) is a better choice than JAX for GPU-accelerated training. Here’s why:
# 	1.	Native Support for Apple Silicon GPUs:
# PyTorch’s MPS backend directly uses the GPU on M2 Pro, making it faster than CPU-bound JAX on macOS.
# 	2.	Simpler Workflow:
# With PyTorch, models can be trained and deployed without needing conversions (e.g., JAX -> TensorFlow -> Core ML).
# 	3.	Versatile Ecosystem:
# PyTorch is widely used, well-documented, and supports almost all ML tasks, including your neural network for QKD.

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
class BB84NN(nn.Module):
    def __init__(self):
        super(BB84NN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization after first layer
        self.fc2 = nn.Linear(64, 32)  # Change 64 -> 32 to match checkpoint
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)  # Change 64 -> 16 to match checkpoint
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 5)  # Change 32 -> 16 to match checkpoint

        self.dropout = nn.Dropout(0.1)  # Dropout to prevent overfitting

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        x = self.dropout(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.1)
        x = self.dropout(p=0.2)

        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.1)
        x = self.dropout(p=0.2)

        x = self.fc4(x)

        # Ensure the output values stay within the defined bounds
        # mu_1 = torch.sigmoid(x[:, 0]) * (1 - 1e-6) + 1e-6
        # mu_2 = torch.sigmoid(x[:, 1]) * (1 - 1e-6) + 1e-6
        # P_mu_1 = torch.sigmoid(x[:, 2]) * (1 - 1e-6) + 1e-6
        # P_mu_2 = torch.sigmoid(x[:, 3]) * (1 - 1e-6) + 1e-6
        # P_X = torch.sigmoid(x[:, 4]) * (1 - 1e-6) + 1e-6

        return x
        # return torch.stack([mu_1, mu_2, P_mu_1, P_mu_2, P_X], dim=1)

# %% [markdown]
# 2. Using the Dataset
# Load and preprocess training_dataset.json
# 
# 	•	L: Represents the fiber length. \
# 	•	n_X: Represents the  n_X  value. \
# 	•	key_rate: Represents the calculated key rate. \
# 	•	optimal_params: A list of 5 optimal parameters.

# %%
# Load dataset
with open('../Training_Data/qkd_dataset_comprehensive_100_20250127_232545.json', 'r') as f:
    data_by_nx = json.load(f)

# Flatten the data structure and filter
cleaned_data = []
for n_x, entries in data_by_nx.items():
    cleaned_data.extend([
        item for item in entries
if item["key_rate"] > 0 and item["e_1"] * 100 <= 200  # Only positive key rates and fiber lengths <= 200 km
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
    P_mu_3 = max(1 - (P_mu_1 + P_mu_2), 1e-6)
    P_Z = max(1 - P_X, 1e-6)
    
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
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform on training data
X_val = scaler.transform(X_val)  # Transform validation data using the same scaler

from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler()  # Scale targets to [0, 1]
Y_train = y_scaler.fit_transform(Y_train)
Y_val = y_scaler.transform(Y_val)

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")

# %% [markdown]
# 4. Training the Model
# 
# Define the training setup:

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
# 

# %%
from tqdm import tqdm

num_epochs = 1000
best_loss = float('inf')  # Initialize best loss to infinity
patience = 20 # Number of epochs to wait before stopping
best_val_loss = float('inf')
early_stop_counter = 0
early_stopping_patience = 400  # Early stopping patience

# Initialize placeholders for final outputs
final_train_loss = None
final_val_loss = None

# Bounds for predicted parameters (adjust as needed)
bounds = [
    (1e-4, 1),  # mu_1
    (1e-4, 1),  # mu_2
    (1e-6, 1),  # P_mu_1
    (1e-6, 1),  # P_mu_2
    (1e-6, 1),  # P_X
]

def compute_loss(predictions, targets, bounds):
    # Calculate standard loss (e.g., MSE)
    loss = criterion(predictions, targets)
    
    # Add penalties for out-of-bounds predictions
    lower_penalties = torch.sum(torch.relu(1e-4 - predictions[:, :2]))  # For mu_1 and mu_2
    upper_penalties = torch.sum(torch.relu(predictions - 1))            # For mu_1 and mu_2
    loss += 0.1 * (lower_penalties + upper_penalties)
    
    return loss

def check_constraints(predictions, bounds):
    """Checks if predictions satisfy the given bounds."""
    for i, (lower, upper) in enumerate(bounds):
        if not (torch.all(predictions[:, i] >= lower) and torch.all(predictions[:, i] <= upper)):
            return False
    return True

max_retries = 10

for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    model.train()
    train_loss = 0  # Track training loss for this epoch

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        
        predictions = model(X_batch.to(device))
        batch_loss = compute_loss(predictions, Y_batch.to(device), bounds=bounds)
    
        batch_loss.backward()
        

        # retry_count = 0
        # while retry_count < max_retries:
        #     # predictions = model(X_batch)
        #     predictions = model(X_batch.to(device))
        #     if check_constraints(predictions, bounds):
        #         break  # Exit retry loop if constraints are satisfied
        #     retry_count += 1
        #     # optimizer.zero_grad()  # Reset optimizer state to retry

        # if retry_count >= max_retries:
        #     print("Warning: Max retries reached, continuing with invalid data")
        
        # Compute the loss
        # batch_loss = criterion(predictions, Y_batch)
        

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += batch_loss.item()

    train_loss /= len(train_loader)

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            predictions = model(X_batch)
            val_loss += criterion(predictions, Y_batch)

    val_loss /= len(val_loader)

    # Store the final loss values
    final_train_loss = train_loss
    final_val_loss = val_loss

    # Learning rate adjustment
    scheduler.step(val_loss)

    # Early stopping

    for epoch in range(num_epochs):
    # Training and validation steps...
    
        current_val_loss = validate()  # Implement a validation function
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print("Early stopping triggered. Training complete.")
            break


# %%
val_loss        

# %%
# Instantiate the corrected model
model = BB84NN()

# Load the saved state dictionary from the checkpoint
checkpoint = torch.load("best_model.pth")

# Load only the model weights (ignoring mismatched keys)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")

# %%
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

# %% [markdown]
# 5. Evaluation and Plotting

# %%
import torch
import numpy as np

# Load the trained model
model = BB84NN().to(device)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define fiber length range and fixed n_X value
fiber_lengths = np.linspace(0.1, 200, 1000)  # Fiber lengths from 0.1 km to 200 km
P_dc_value = 6e-7  # Example value (adjust as needed)
e_mis = 5e-3       # Example misalignment error (adjust as needed)
target_nx = 1e13   # Fixed n_X value

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
for i in range(5):
    print(f"Fiber Length {fiber_lengths[i]:.1f} km -> {predicted_params_array[i]}")

# %%
for i in range(len(predicted_params_list)):
    mu_1, mu_2, P_mu_1, P_mu_2, P_X = predicted_params_list[i]
    
    # Apply constraint corrections
    mu_3 = 2e-4
    P_mu_3 = max(1 - (P_mu_1 + P_mu_2), 1e-6)  # Ensure non-negative probability
    P_Z = max(1 - P_X, 1e-6)

    predicted_params_list[i] = [mu_1, mu_2, P_mu_1, P_mu_2, P_X, P_mu_3, P_Z]

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

# Load the trained model correctly
model = BB84NN().to(device)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Set n_X = 10^12
target_nx = 1e12
e_4 = np.log10(target_nx)

# Generate fiber lengths
fiber_lengths = np.linspace(0.1, 200, 1000)

# Lists to store all results
predicted_key_rates = []
predicted_params_list = []
eta_ch_list = []
S_X_0_list = []
S_Z_0_list = []
S_X_1_list = []
S_Z_1_list = []
tau_0_list = []
tau_1_list = []
e_mu_k_list = []
e_obs_X_list = []
v_Z_1_list = []
gamma_list = []
Phi_X_list = []
binary_entropy_Phi_list = []
lambda_EC_list = []
l_calculated_list = []

# For each fiber length
for L in fiber_lengths:
    # Prepare input for neural network
    e_1 = L / 100
    e_2 = -np.log10(P_dc_value)
    e_3 = e_mis * 100
    
    # Neural network input
    X = torch.tensor([[e_1, e_2, e_3, e_4]], dtype=torch.float32).to(device)
    
    # Get predicted parameters
    with torch.no_grad():
        params = model(X).cpu().numpy()[0]
        predicted_params_list.append(params)
    
    # Calculate all values using objective function
    result = objective(
        params,  # predicted parameters
        L,       # current fiber length
        target_nx,
        alpha, eta_Bob, P_dc_value, 
        epsilon_sec, epsilon_cor, f_EC, 
        e_mis, P_ap, n_event
    )
    
    # Unpack all results
    (penalized_key_rate, eta_ch, S_X_0, S_Z_0, S_X_1, S_Z_1, 
     tau_0, tau_1, e_mu_k, e_obs_X, v_Z_1, gamma, Phi_X,
     binary_entropy_Phi, lambda_EC, l_calculated) = result
    
    # Store all results
    predicted_key_rates.append(penalized_key_rate)
    eta_ch_list.append(eta_ch)
    S_X_0_list.append(S_X_0)
    S_Z_0_list.append(S_Z_0)
    S_X_1_list.append(S_X_1)
    S_Z_1_list.append(S_Z_1)
    tau_0_list.append(tau_0)
    tau_1_list.append(tau_1)
    e_mu_k_list.append(e_mu_k)
    e_obs_X_list.append(e_obs_X)
    v_Z_1_list.append(v_Z_1)
    gamma_list.append(gamma)
    Phi_X_list.append(Phi_X)
    binary_entropy_Phi_list.append(binary_entropy_Phi)
    lambda_EC_list.append(lambda_EC)
    l_calculated_list.append(l_calculated)

# Convert to numpy arrays
predicted_params_array = np.array(predicted_params_list)

# Create multiple subplots for all the metrics
plt.figure(figsize=(20, 25))

# Plot 1: Key Rate
plt.subplot(5, 2, 1)
plt.plot(fiber_lengths, predicted_key_rates, 'b-', label='Key Rate')
plt.xlabel('Fiber Length (km)')
plt.ylabel('Key Rate')
plt.yscale('log')
plt.title('Predicted Key Rate')
plt.grid(True)
plt.legend()

# Plot 2: Parameters
plt.subplot(5, 2, 2)
param_names = ['mu_1', 'mu_2', 'P_mu_1', 'P_mu_2', 'P_X_value']
for i in range(5):
    plt.plot(fiber_lengths, predicted_params_array[:, i], label=param_names[i])
plt.xlabel('Fiber Length (km)')
plt.ylabel('Parameter Value')
plt.title('Predicted Parameters')
plt.grid(True)
plt.legend()

# Plot additional metrics
metrics = [
    (eta_ch_list, 'Channel Transmittance'),
    (S_X_0_list, 'S_X_0'),
    (S_Z_0_list, 'S_Z_0'),
    (tau_0_list, 'tau_0'),
    (e_obs_X_list, 'e_obs_X'),
    (v_Z_1_list, 'v_Z_1'),
    (gamma_list, 'gamma'),
    (l_calculated_list, 'l_calculated')
]

for i, (metric, title) in enumerate(metrics, 3):
    plt.subplot(5, 2, i)
    plt.plot(fiber_lengths, metric, 'g-')
    plt.xlabel('Fiber Length (km)')
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True)

plt.tight_layout()
plt.savefig('comprehensive_predictions_nx_1e12.png')
plt.show()

# Print some values at specific distances
distances = [0,20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
print("\nPredicted Values at Specific Distances:")
print("Distance (km) | Key Rate | eta_ch | S_X_0 | e_obs_X")
print("-" * 60)
for d in distances:
    idx = np.abs(fiber_lengths - d).argmin()
    print(f"{d:11.0f} | {predicted_key_rates[idx]:8.2e} | {eta_ch_list[idx]:6.4f} | {S_X_0_list[idx]:6.4f} | {e_obs_X_list[idx]:6.4f}")



# %%
# Step 2: Load and Plot Dataset
def plot_for_nx(data, target_nx):
    """
    Plot results for a specific n_X value
    """
    # Convert target_nx to string for dictionary key
    nx_key = str(float(target_nx))
    
    if nx_key not in data:
        print(f"No data found for n_X = {target_nx}")
        return
        
    filtered_data = data[nx_key]
    
    if not filtered_data:
        print(f"No valid results for n_X = {target_nx}")
        return
    
    # Extract data
    fiber_lengths = [entry["fiber_length"] for entry in filtered_data]
    key_rates = [entry["key_rate"] for entry in filtered_data]
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Plot key rates
    plt.subplot(1, 2, 1)
    plt.plot(fiber_lengths, np.log10([max(kr, 1e-30) for kr in key_rates]))
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("log10(Key Rate)")
    plt.title(f"Key Rate vs Fiber Length (n_X = {target_nx:.0e})")
    plt.grid(True)
    
    # Plot parameters
    plt.subplot(1, 2, 2)
    params = ["mu_1", "mu_2", "P_mu_1", "P_mu_2", "P_X_value"]
    for param in params:
        values = [entry["optimized_params"][param] for entry in filtered_data]
        plt.plot(fiber_lengths, values, label=param)
    
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("Parameter Value")
    plt.title(f"Optimized Parameters (n_X = {target_nx:.0e})")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"qkd_results_nx_{target_nx:.0e}.png", dpi=300, bbox_inches="tight")
    plt.show()

# # Load the saved dataset
# with open("qkd_dataset_high_nx.json", 'r') as f:
#     dataset = json.load(f)

# Load the saved dataset
with open("../Training_Data/qkd_dataset_comprehensive_20250122_004748.json", 'r') as f:
    dataset = json.load(f)

# Plot results for each n_X value
for nx in [10**s for s in range(11, 15, 1)]:
    print(f"\nPlotting results for n_X = {nx:.0e}")
    plot_for_nx(dataset, nx)

# %%
# Load and filter data for specific n_X (10^12)
with open("../Training_Data/qkd_dataset_comprehensive_20250122_004748.json", "r") as f:
    dataset = json.load(f)

target_nx = "1e12"  # or "1000000000000.0" depending on your data format
filtered_data = []

# Handle the dictionary structure where n_X values are keys
if isinstance(dataset, dict):
    # Get data for specific n_X
    for n_x_key, entries in dataset.items():
        if float(n_x_key) == 1e12:  # Check for 10^12
            filtered_data.extend(entries)

print(f"Number of entries for n_X = 10^12: {len(filtered_data)}")

# Plot comparison for n_X = 10^12
plt.figure(figsize=(12, 8))

# Extract data for plotting
fiber_lengths = [entry['fiber_length'] for entry in filtered_data]
optimal_key_rates = [entry['key_rate'] for entry in filtered_data]

# Get predictions for the same fiber lengths
test_inputs = []
for entry in filtered_data:
    test_inputs.append([
        entry['e_1'],
        entry['e_2'],
        entry['e_3'],
        entry['e_4']
    ])

# Convert to tensor and get predictions
test_tensor = torch.tensor(test_inputs, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    predictions = model(test_tensor).cpu().numpy()

# Plot results
plt.subplot(2, 1, 1)
plt.plot(fiber_lengths, np.log10(optimal_key_rates), 'b-', label='Optimal Key Rate')
plt.plot(fiber_lengths, np.log10(predictions[:, 0]), 'r--', label='Predicted Key Rate')
plt.xlabel('Fiber Length (km)')
plt.ylabel('log10(Key Rate)')
plt.title('Key Rate Comparison for n_X = 10^12')
plt.legend()
plt.grid(True)

# Plot parameters
plt.subplot(2, 1, 2)
param_names = ['mu_1', 'mu_2', 'P_mu_1', 'P_mu_2', 'P_X_value']
optimal_params = np.array([list(entry['optimized_params'].values()) for entry in filtered_data])
predicted_params = predictions[:, 1:]  # Skip key rate column

for i in range(5):
    plt.plot(fiber_lengths, optimal_params[:, i], '-', label=f'Optimal {param_names[i]}')
    plt.plot(fiber_lengths, predicted_params[:, i], '--', label=f'Predicted {param_names[i]}')

plt.xlabel('Fiber Length (km)')
plt.ylabel('Parameter Value')
plt.title('Parameter Comparison for n_X = 10^12')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print some statistics
print("\nStatistics for n_X = 10^12:")
mse_key_rate = np.mean((np.log10(optimal_key_rates) - np.log10(predictions[:, 0]))**2)
print(f"MSE for log10(Key Rate): {mse_key_rate:.6f}")

for i in range(5):
    mse_param = np.mean((optimal_params[:, i] - predicted_params[:, i])**2)
    print(f"MSE for {param_names[i]}: {mse_param:.6f}")

# %%


# %%

# Load the training dataset for the optimal key rates and parameters

# Load the dataset
with open("../Training_Data/total_training_dataset.json", "r") as f:
    dataset = json.load(f)

# Filter for n_X = 10^6 (e_4 = log10(10^6) = 6)
filtered_data = [entry for entry in dataset if np.isclose(entry["e_4"], 6, atol=1e-2)]

print(f"Number of entries for n_X = 10^6: {len(filtered_data)}")

# Extract fiber lengths and optimized parameters
Ls_training = np.array([item["e_1"] * 100 for item in training_data])  # Convert normalized to km
optimal_params = np.array([item["optimized_params"] for item in training_data])  # Extract optimal parameters

# Load the best model
model.load_state_dict(torch.load("best_model.pth")['model_state_dict'])
model.eval()

# Make predictions on validation data

with torch.no_grad():
    predicted_Y = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()

    # Debugging: Check a few sample predictions
    sample_predictions = model(torch.tensor(X_val[:10], dtype=torch.float32).to(device)).cpu().numpy()
    print("Sample Predictions:\n", sample_predictions)
    print("Sample Ground Truth:\n", Y_val[:10])

# Extract distances (fiber lengths) from validation set
distances_km = X_val[:, 0] * 100  # Convert normalized distance back to km

# Convert normalized e_1 back to kilometers and clip negatives
distances_km = np.clip(X_val[:, 0] * 100, a_min=0, a_max=None)  # Ensure no negative distances
Ls_training = np.clip([item["e_1"] * 100 for item in training_data], a_min=0, a_max=None)

plt.figure(figsize=(10, 6))
# Plot the line for the optimal key rate from the training dataset
plt.plot(Ls, np.log10(optimal_key_rates), 'b-', label="Optimal (Training Dataset)")

# Plot the predicted key rates for validation data
plt.scatter(distances_km, np.log10(key_rate_predicted), color='r', label="NN Predicted", alpha=0.5)

plt.xlabel("Fiber Length (km)")
plt.ylabel("log_{10}(Key Rate)")
plt.title("Key Rate vs Distance")
plt.legend()
plt.grid(True)
plt.show()

# ---- Plot 2: Parameter Comparison ----
plt.figure(figsize=(10, 6))
# Extract optimized parameters from the training data
optimized_params = np.array([list(item["optimized_params"].values()) for item in training_data], dtype=np.float32)

plt.figure(figsize=(10, 6))

for i in range(5):  # For each parameter
    # Extract parameter values for plotting
    optimized_values = optimized_params[:, i]  # Optimal parameters from training data
    predicted_values = predicted_Y[:, i]  # Predicted parameters from NN

    # Plot the optimal parameter as a line across all distances
    plt.plot(Ls_training, optimized_values, label=f"Optimized Param {i+1}", linestyle='--')

    # Plot the predicted parameter as points across validation distances
    plt.scatter(distances_km, predicted_values, label=f"NN Predicted Param {i+1}", alpha=0.5)

plt.xlabel("Fiber Length (km)")
plt.ylabel("Parameter Value")
plt.title("Predicted vs Optimal Parameters Across Distance")
plt.legend()
plt.grid(True)
plt.show()


