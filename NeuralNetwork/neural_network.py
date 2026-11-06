# %%
# %% [markdown]
# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import json
from tqdm import tqdm
import torch.nn.functional as F
import joblib
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
from torch.utils.data import DataLoader, random_split
import time  # For timing
from scipy.optimize import dual_annealing
import pandas as pd

# Set matplotlib to use a non-interactive backend
plt.switch_backend('Agg')

# PhD-Grade Presentation Aesthetics
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 3,
    'lines.markersize': 10,
    'figure.titlesize': 26,
    'figure.dpi': 300,
    'grid.alpha': 0.4
})

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# The project root is one level up from NeuralNetwork/
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
# Also set notebook_dir for later pathing consistency
notebook_dir = script_dir


# %% [markdown]
# Checking if the MPS (Metal Performance Shaders) backend is available

# %%
# Check if MPS is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS device not available. Check if PyTorch and macOS set up correctly.")

# Set the device to MPS
device = torch.device("mps")  # Use GPU on M2 Pro

# %%
# %%
# Import the objective function
# Import the physics engine from the src folder
from src.qkd.physics import (
    calculate_eta_ch, calculate_eta_sys, calculate_D_mu_k, 
    calculate_n_X_total, calculate_N, calculate_n_Z_total,
    calculate_e_mu_k, calculate_e_obs, calculate_h, calculate_lambda_EC,
    calculate_sqrt_term, calculate_n_pm, calculate_S_0, calculate_S_1,
    calculate_m_mu_k, calculate_m_pm, calculate_v_1, calculate_gamma,
    calculate_Phi, calculate_l, calculate_R
)
from src.qkd.model import calculate_key_rates_and_metrics, objective, penalty


# %%
# Load dataset
# Direct path to the staged high-fidelity Hessian dataset
latest_file = os.path.join(notebook_dir, 'Training_Data/reordered_qkd_grouped_dataset_20260217_014536.json')
print(f"📂 Loading staged Hessian dataset: {latest_file}")

with open(latest_file, 'r') as f:
    data_by_nx = json.load(f)


print(f"The overall dataset contains {len(data_by_nx)} entries (number of unique n_X values).")

# Print the number of entries for each n_X before filtering
for n_x in data_by_nx.keys():
    print(f"Number of entries for n_X = {n_x}: {len(data_by_nx[n_x])}")


# %%
# Flatten the data structure and filter
cleaned_data = []
for n_x, entries in data_by_nx.items():
    filtered_entries = [item for item in entries if item["key_rate"] > 0 and item["e_1"] * 100 <= 200]
    print(f"After filtering, n_X = {n_x} has {len(filtered_entries)} entries.")
    cleaned_data.extend(filtered_entries)

# Verify the cleaned dataset
if not cleaned_data:
    print("No valid data after filtering.")
else:
    print(f"Filtered dataset contains {len(cleaned_data)} entries.")
    print("\nSample entry from the cleaned dataset:")
    print(json.dumps(cleaned_data[0], indent=2))
    print("\nNumber of unique n_X values:", len(data_by_nx))

# %%
X = np.array([[item['e_1'], item['e_2'], item['e_3'], item['e_4']] for item in cleaned_data])
# Step 1: Log-Target Transformation (predicting log10(SKR))
Y = np.array([[
    item['optimized_params']['mu_1'], 
    item['optimized_params']['mu_2'], 
    item['optimized_params']['P_mu_1'], 
    item['optimized_params']['P_mu_2'], 
    item['optimized_params']['P_X_value'],
    np.log10(np.maximum(item['key_rate'], 1e-20)) 
] for item in cleaned_data])

# Step 2: Weighted Loss for n_X = 10^4 and 10^5 (e_4 <= 5)
# Increase 10^4 weight to 10.0 for extreme fidelity at the "Security Cliff"
weights = np.array([10.0 if item['e_4'] <= 4.1 else (5.0 if item['e_4'] <= 5.1 else 1.0) for item in cleaned_data])

# %%
from sklearn.utils import shuffle

X, Y, weights = shuffle(X, Y, weights, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)  # Fit and transform on training data

y_scaler = MinMaxScaler()  # Scale targets to [0, 1]
Y = y_scaler.fit_transform(Y)

# Save the scalers
os.makedirs(os.path.join(notebook_dir, 'models'), exist_ok=True)
joblib.dump(scaler, os.path.join(notebook_dir, 'models/scaler.pkl'))  # Save StandardScaler
joblib.dump(y_scaler, os.path.join(notebook_dir, 'models/y_scaler.pkl'))  # Save MinMaxScaler


print(f"X shape: {X.shape}, Y shape: {Y.shape}")
dataset = TensorDataset(
    torch.tensor(X, dtype=torch.float32), 
    torch.tensor(Y, dtype=torch.float32),
    torch.tensor(weights, dtype=torch.float32)
)

# %%
# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders with increased batch size for GPU efficiency
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

# %% [markdown]
# ### Prepare Evaluation Data for All n_X Values (10^4 to 10^9) Before Training

# Load the combined dataset
combined_data = data_by_nx
print(f"Using current dataset with {len(combined_data)} n_X entries for evaluation.")


# %%
# Define the range of n_X values to evaluate
nx_values = [10**s for s in range(4, 10)]  # 10^4 to 10^9

# Dictionary to store evaluation data for each n_X
all_evaluation_data = {}

# Physics constants used across all optimization and evaluation
PHYSICS_CONSTANTS = {
    'alpha': 0.2, 
    'eta_Bob': 0.1, 
    'P_dc_value': 6e-7, 
    'epsilon_sec': 1e-10, 
    'epsilon_cor': 1e-15, 
    'f_EC': 1.16, 
    'e_mis': 5e-3, 
    'P_ap': 0, 
    'n_event': 1
}

# %%
# Define a safe wrapper for the objective function
def safe_objective(params, L, nx, **kwargs):
    # Merge provided kwargs with defaults
    full_kwargs = {**PHYSICS_CONSTANTS, **kwargs}
    try:
        # Ensure params are in a numpy array for JAX ravel compatibility
        params_np = np.array(params)
        if np.any(np.isnan(params_np)) or np.any(np.isinf(params_np)):
             return None

        # Call the JAX objective function (returns a tuple, key_rate is at index 0)
        key_rate = objective(params_np, L, nx, **full_kwargs)[0]

        if np.isnan(key_rate) or np.isinf(key_rate) or key_rate <= 0:
            return None 

        return float(key_rate)

    except Exception as e:
        return None

# %%
# Extract and prepare evaluation data for each n_X
for nx in nx_values:
    nx_key = str(float(nx))
    if nx_key not in combined_data:
        print(f"No data found for n_X = {nx} in the combined dataset. Skipping...")
        continue
    evaluation_data = combined_data[nx_key]
    print(f"Extracted {len(evaluation_data)} entries for n_X = {nx}.")

    # Extract fiber lengths and optimized parameters
    fiber_lengths = np.array([entry["fiber_length"] for entry in evaluation_data])
    optimized_params_array = np.array([list(entry["optimized_params"].values()) for entry in evaluation_data])

    # Compute optimized key rates
    optimized_key_rates = []
    valid_indices = []
    for idx, (params, L) in enumerate(zip(optimized_params_array, fiber_lengths)):
        key_rate = safe_objective(params, L, nx, **PHYSICS_CONSTANTS)
        if key_rate is None:
            continue
        optimized_key_rates.append(key_rate)
        valid_indices.append(idx)
    optimized_key_rates = np.array(optimized_key_rates)

    if len(valid_indices) == 0:
        continue

    fiber_lengths = fiber_lengths[valid_indices]
    optimized_params_array = optimized_params_array[valid_indices]

    # Prepare test inputs for evaluation
    X_test = []
    for L in fiber_lengths:
        e_1 = L / 100
        e_2 = -np.log10(6e-7)
        e_3 = 5e-3 * 100
        e_4 = np.log10(nx)
        X_test.append([e_1, e_2, e_3, e_4])
    X_test = np.array(X_test)

    if X_test.size == 0:
         continue
        
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # Prepare full 200km evaluation grid for Student prediction (Decoupled from Teacher)
    X_test_full = []
    full_fiber_lengths = np.linspace(1, 200, 200) # 1km to 200km
    for L in full_fiber_lengths:
        e_1 = L / 100
        e_2 = -np.log10(6e-7)
        e_3 = 5e-3 * 100
        e_4 = np.log10(nx)
        X_test_full.append([e_1, e_2, e_3, e_4])
    X_test_full = np.array(X_test_full)
    X_test_full_scaled = scaler.transform(X_test_full)
    X_test_full_tensor = torch.tensor(X_test_full_scaled, dtype=torch.float32).to(device)

    # Store all evaluation data in a dictionary
    all_evaluation_data[nx] = {
        'fiber_lengths': fiber_lengths,
        'optimized_params_array': optimized_params_array,
        'optimized_key_rates': optimized_key_rates,
        'X_test_tensor': X_test_tensor,
        'X_test_full_tensor': X_test_full_tensor, # NEW: For full 200km plotting
        'full_fiber_lengths': full_fiber_lengths  # NEW: For full 200km plotting
    }
    print(f"After filtering invalid key rates, {len(optimized_key_rates)} data points remain for n_X = {nx}.")

# %%
# Define the neural network model
class BB84NN(nn.Module):
    def __init__(self):
        super(BB84NN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 6) # Predict 5 params + 1 log10(SKR)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# %%
# Initialize model, loss, optimizer, and scheduler
model = BB84NN().to(device)
criterion = nn.MSELoss()
# Step 3: Add L2 Regularization (Weight Decay)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

# Lists to store metrics
train_losses = []
val_losses = []
learning_rates = []

# %%
def plot_relative_errors(fiber_lengths, optimized_key_rates, optimized_params_array, 
                         predicted_key_rates, predicted_params_array, epoch, filename, 
                         nx=None, threshold=1e-7):
    # Filter out small key rates for relative error stability - DISABLED for low n_X visibility
    # valid_mask = ~(np.isnan(optimized_key_rates) | (optimized_key_rates <= threshold))
    # fiber_lengths = fiber_lengths[valid_mask]
    # optimized_key_rates = optimized_key_rates[valid_mask]
    # predicted_key_rates = predicted_key_rates[valid_mask]
    # optimized_params_array = optimized_params_array[valid_mask]
    # predicted_params_array = predicted_params_array[valid_mask]

    if len(fiber_lengths) == 0: return

    # Compute relative errors
    relative_errors = []
    param_labels = ['$\mu_1$', '$\mu_2$', '$P_{\mu_1}$', '$P_{\mu_2}$', '$P_X$']
    for i in range(5):
        optimized = optimized_params_array[:, i]
        predicted = predicted_params_array[:, i]
        denominator = np.maximum(optimized, 1e-10)
        rel_error = (predicted - optimized) / denominator
        relative_errors.append(rel_error)
    
    key_rate_rel_error = (predicted_key_rates - optimized_key_rates) / np.maximum(optimized_key_rates, threshold)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    for i in range(5):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.plot(fiber_lengths, relative_errors[i], 'b-', linewidth=2.5, label=f'Rel Error {param_labels[i]}')
        ax.set_xlabel('Fiber Length (km)')
        ax.set_title(f'Rel Error: {param_labels[i]}', fontsize=20)
        ax.set_title(f'Rel Error: {param_labels[i]}', fontsize=20)
        ax.grid(True)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 200) # Ensure full range is shown

    ax_key = fig.add_subplot(gs[1, 2])
    ax_key.plot(fiber_lengths, key_rate_rel_error, 'r-', linewidth=2.5, label='Rel Error Key Rate')
    ax_key.set_xlabel('Fiber Length (km)')
    ax_key.set_title('Rel Error: Key Rate', fontsize=20)
    ax_key.grid(True)
    ax_key.set_ylim(-1, 1)

    for ax in fig.get_axes():
        ax.set_xlim(0, 200) # Explicitly set x-limit to 200km per user request

    exponent = int(np.log10(nx)) if nx is not None else ''
    fig.suptitle(f'Relative Errors for $n_X = 10^{{{exponent}}}$', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_keyrate_subplots(plot_data, epoch, filename):
    nx_vals = sorted(plot_data.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, nx in enumerate(nx_vals):
        data = plot_data[nx]
        ax = axes[i]
        
        # Show cases over 150km: Apply a floor of 1e-15 to show the "Security Cliff"
        gt_rates = np.maximum(data['optimized_key_rates'], 1e-15)
        # For the student, we use prediction which might be valid or safe_objective might have returned None
        # We align them over the full fiber_lengths range
        ax.semilogy(data['fiber_lengths'], gt_rates, 'b-', linewidth=4, label='Hessian Teacher')
        
        # We need to make sure predicted_key_rates matches fiber_lengths_pred
        # If lengths are different, we plot them separately but they share the same x-range
        ax.semilogy(data['fiber_lengths_pred'], np.maximum(data['predicted_key_rates'], 1e-15), 'r--', linewidth=3, label='Neural Student')
        
        ax.set_title(f'$n_X = 10^{{{int(np.log10(nx))}}}$ (Full 200km Horizon)', pad=15)
        ax.set_xlabel('Fiber Length (km)')
        ax.set_ylabel('Key Rate (bits/pulse)')
        ax.set_xlim(0, 200)
        ax.set_ylim(1e-15, 1e-2)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')
    plt.suptitle(f"Multi-Regime Secret Key Rate Inheritance ({epoch} Epochs Convergence)", y=0.98, fontsize=30)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Key rate subplots saved to {filename}")

def plot_parameters_subplots(plot_data, epoch, filename):
    nx_vals = sorted(plot_data.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    colors = ['red', 'purple', 'orange', 'green', 'blue']
    labels = ['$\mu_1$', '$\mu_2$', '$P_{\mu_1}$', '$P_{\mu_2}$', '$P_X$']
    for i, nx in enumerate(nx_vals):
        data = plot_data[nx]
        ax = axes[i]
        for j in range(5):
            ax.plot(data['fiber_lengths'], data['optimized_params_array'][:, j], color=colors[j], linewidth=3, alpha=0.3)
            ax.plot(data['fiber_lengths_pred'], data['predicted_params_array'][:, j], color=colors[j], linewidth=2, linestyle='--', label=labels[j] if i==0 else "")
        ax.set_title(f'$n_X = 10^{{{int(np.log10(nx))}}}$ (Optimized Parameters)', pad=15)
        ax.set_xlabel('Fiber Length (km)')
        ax.set_ylabel('Param Value')
        ax.set_ylim(0, 1)
        ax.grid(True)
    fig.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02), frameon=True)
    plt.suptitle(f"Multi-Regime Parameter Strategy Inheritance ({epoch} Epochs Fidelity)", y=1.05, fontsize=30)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Parameter subplots saved to {filename}")

# %%
# Training loop
total_start_time = time.time()
num_epochs = 5000
print(f"🚀 Starting training for {num_epochs} epochs (Fidelity Focus)...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets, item_weights in train_loader:
        inputs, targets, item_weights = inputs.to(device), targets.to(device), item_weights.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Apply sample weights
        loss = (F.mse_loss(outputs, targets, reduction='none') * item_weights.unsqueeze(1)).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, targets, item_weights in val_loader:
            inputs, targets, item_weights = inputs.to(device), targets.to(device), item_weights.to(device)
            outputs = model(inputs)
            loss = (F.mse_loss(outputs, targets, reduction='none') * item_weights.unsqueeze(1)).mean()
            val_running_loss += loss.item() * inputs.size(0)
    val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr) # Ensure tracking
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6e}")

total_training_time = time.time() - total_start_time
print(f"--- Total Training Time: {total_training_time:.2f} seconds ---")

# Save model
torch.save(model.state_dict(), os.path.join(notebook_dir, 'models/bb84_nn_model.pth'))
print("Model saved to models/bb84_nn_model.pth")

# %% [markdown]
# ### 5. Final Evaluation and Asset Generation
output_assets = os.path.join(project_root, 'assets')
os.makedirs(output_assets, exist_ok=True)

# 1. Performance Diagnostics: Relative Error Plots for all n_X
print("\n--- Generating Relative Error Plots and Subplots ---")
plot_data = {}
if 'all_evaluation_data' in locals():
    # Compute predictions for the FULL 200km grid for plotting
    for nx_target, data in all_evaluation_data.items():
        with torch.no_grad():
            # 1. Prediction on VALIDATED points (for metrics like relative error)
            outputs_s = model(data['X_test_tensor']).cpu().numpy()
            outputs_a = y_scaler.inverse_transform(np.clip(outputs_s, 0, 1))
            pred_p = outputs_a[:, :5]
            # ... (Physics constraints on pred_p same as before) ...
            pred_p[:, 0] = np.maximum(pred_p[:, 0], 1e-6)
            pred_p[:, 1] = np.maximum(pred_p[:, 1], 1e-6)
            pred_p[:, 2:5] = np.clip(pred_p[:, 2:5], 0, 1)
            p_sum = pred_p[:, 2] + pred_p[:, 3]
            mask = p_sum > 1.0
            if np.any(mask):
                pred_p[mask, 2] /= p_sum[mask]
                pred_p[mask, 3] /= p_sum[mask]

            p_ks, v_idx = [], []
            for idx, (p, L) in enumerate(zip(pred_p, data['fiber_lengths'])):
                k = safe_objective(p, L, nx_target, **PHYSICS_CONSTANTS)
                if k is not None:
                    p_ks.append(k)
                    v_idx.append(idx) # Keep valid skrs for skr error
                else:
                    p_ks.append(np.nan) # Append nan to keep length same
                    # v_idx.append(idx) # ALSO APPEND to v_idx to show parameter error even if key rate is 0
            
            # Use ALL points for parameter relative error plotting
            v_idx_params = list(range(len(pred_p))) 
            
            # 2. Prediction on FULL 200km grid (for plotting curves)
            outputs_full_s = model(data['X_test_full_tensor']).cpu().numpy()
            outputs_full_a = y_scaler.inverse_transform(np.clip(outputs_full_s, 0, 1))
            pred_full_p = outputs_full_a[:, :5]
            pred_full_skr = np.maximum(10**outputs_full_a[:, 5], 1e-20) # log-detransform
            
            # Store full predictions for plotting
            plot_data[nx_target] = {
                'fiber_lengths': data['fiber_lengths'],
                'optimized_key_rates': data['optimized_key_rates'],
                'optimized_params_array': data['optimized_params_array'],
                'predicted_key_rates': pred_full_skr,
                'fiber_lengths_pred': data['full_fiber_lengths'],
                'predicted_params_array': pred_full_p
            }

            if len(v_idx_params) > 0:
                exp = int(np.log10(nx_target))
                fname = f'parameter_relative_error_nx_1e{exp:02d}.png'
                plot_relative_errors(
                    data['fiber_lengths'], # Use FULL data
                    data['optimized_key_rates'], 
                    data['optimized_params_array'],
                    np.array(p_ks), pred_p, # Use FULL pred_p
                    num_epochs, os.path.join(output_assets, fname), nx=nx_target
                )
    
    # Generate multi-regime subplots using the FULL data
    if plot_data:
        plot_keyrate_subplots(plot_data, num_epochs, os.path.join(output_assets, 'keyrate_subplots_last_epoch.png'))
        plot_parameters_subplots(plot_data, num_epochs, os.path.join(output_assets, 'parameters_subplots_last_epoch.png'))

# 2. Stability Analysis
print("\n--- Generating Stability Comparison (Real Data) ---")
# Load JAX (Hessian) Data
with open(latest_file, 'r') as f:
    data_hessian = json.load(f)

# Load Pure DA Data
pure_da_path = os.path.join(project_root, 'Optimization/run_scripts/outputs/JAX/Pure_DA/run_PURE_DA_20260216_003022/qkd_grouped_dataset_20260216_003022.json')
try:
    with open(pure_da_path, 'r') as f:
        data_da = json.load(f)
    print(f"Loaded Pure DA data from {pure_da_path}")
except FileNotFoundError:
    print("Warning: Pure DA data not found. Using Hessian data as placeholder for comparison structure.")
    data_da = data_hessian

nx_s = "1000000000.0"
if nx_s in data_hessian and nx_s in data_da:
    # Extract Hessian Data
    entries_h = sorted(data_hessian[nx_s], key=lambda x: x['fiber_length'])
    f_len_h = np.array([e['fiber_length'] for e in entries_h])
    p_h = np.array([e['optimized_params']['P_mu_1'] for e in entries_h])
    
    # Extract DA Data (and interpolate to match Hessian fiber lengths for plotting)
    entries_d = sorted(data_da[nx_s], key=lambda x: x['fiber_length'])
    f_len_d = np.array([e['fiber_length'] for e in entries_d])
    p_d_raw = np.array([e['optimized_params']['P_mu_1'] for e in entries_d])
    
    # Interpolate DA to Hessian grid for clean comparison
    p_d_interp = np.interp(f_len_h, f_len_d, p_d_raw)
    
    plt.figure(figsize=(10, 6))
    plt.plot(f_len_h, p_d_interp, 'r--', label='Global Search (Dual Annealing)', alpha=0.6, linewidth=1)
    plt.plot(f_len_h, p_h, 'b-', label='Physics-Aware (Hessian)', linewidth=2.5)
    
    plt.title(f'Optimization Stability Comparison ($P_{{\mu_1}}$ @ $n_X=10^9$)')
    plt.xlabel('Fiber Length (km)'); plt.ylabel('Parameter Value ($P_{{\mu_1}}$)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 200); plt.ylim(0, 1)
    plt.savefig(os.path.join(output_assets, 'optimization_stability_comparison.png'), dpi=300)
    print("✓ Stability comparison plot saved.")

# 3. Dynamic vs Static
from src.qkd.model import calculate_key_rates_and_metrics
print("\n--- Generating Dynamic vs Static Overlay ---")
static_p = np.array([0.6, 0.1, 0.5, 0.1, 0.5]) 
static_L = np.linspace(1, 200, 50)
static_rates = calculate_key_rates_and_metrics(static_p, static_L, 1e9, 
                                            alpha=0.2, eta_Bob=0.1, P_dc_value=6e-7, 
                                            epsilon_sec=1e-10, epsilon_cor=1e-15, f_EC=1.16, 
                                            e_mis=5e-3, P_ap=0, n_event=1)[0]
dyn_f = [e['fiber_length'] for e in entries if e['fiber_length'] <= 200]
dyn_r = [e['key_rate'] for e in entries if e['fiber_length'] <= 200]
plt.figure(figsize=(10, 6))
plt.semilogy(static_L, static_rates, 'r--', label='Static Config')
plt.semilogy(dyn_f, dyn_r, 'b-', label='AI-Optimized (Dynamic)')
plt.title('Secret Key Rate: Dynamic (AI) vs Static')
plt.xlabel('Fiber Length (km)'); plt.ylabel('Key Rate (bits/pulse)'); plt.legend(); plt.grid(True); plt.ylim(1e-10, 1e-2)
plt.savefig(os.path.join(output_assets, 'Dynamic_vs_Static_Overlay.png'), dpi=300)

# 4. Sensitivity Analysis (CORRECTED)
from src.qkd.model import objective_val_and_grad
print("\n--- Running Sensitivity Analysis ---")

# Use the exact same physics constants as the rest of the script!
full_bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 1.0), (1e-12, 1.0), (1e-12, 1.0)]

def get_s_opt(idx=None, val=None):
    def obj_s(p):
        p_f = list(p)
        if idx is not None: 
            p_f.insert(idx, val)
            
        # Optional: Hard constraint penalty so probabilities don't exceed 1.0
        if p_f[2] + p_f[3] > 1.0:
            return 1e9 

        v, _ = objective_val_and_grad(np.array(p_f), 100.0, 1e8, **PHYSICS_CONSTANTS)
        return float(v)
        
    if idx is not None:
        active_bounds = full_bounds[:idx] + full_bounds[idx+1:]
    else:
        active_bounds = full_bounds
        
    # maxiter bumped to 1000 to ensure convergence and prevent negative artifacts
    res = dual_annealing(obj_s, bounds=active_bounds, maxiter=1000)
    return -res.fun, res.x

b_r, b_p = get_s_opt()
p_n = ['Signal ($\mu_1$)', 'Decoy ($\mu_2$)', 'Prob ($P_{\mu_1}$)', 'Prob ($P_{\mu_2}$)', 'Basis ($P_X$)']
s_r = []

for i, name in enumerate(p_n):
    o_v = b_p[i]
    param_min, param_max = full_bounds[i]
    
    # Calculate exact 20% shifts, safely clamped to the specific parameter's true bounds
    val_high = min(o_v * 1.2, param_max)
    val_low = max(o_v * 0.8, param_min)
    
    r_h, _ = get_s_opt(i, val_high)
    r_l, _ = get_s_opt(i, val_low)
    
    # Calculate worst-case percentage drop (floor at 0 to prevent artifacts if DA still slightly jitters)
    drop_high = max((b_r - r_h) / b_r, 0)
    drop_low = max((b_r - r_l) / b_r, 0)
    
    s_r.append({'Parameter': name, 'Sensitivity (%)': max(drop_high, drop_low) * 100})

pd.DataFrame(s_r).sort_values(by='Sensitivity (%)').plot.barh(x='Parameter', y='Sensitivity (%)', color='#ff7f0e', figsize=(10, 6))
plt.title('Parameter Sensitivity (L=100km, $n_X=10^8$)')
plt.savefig(os.path.join(output_assets, 'parameter_sensitivity.png'), dpi=300)

# 5. Final Performance Diagram
print("\n--- Generating Final Performance Diagram (n_X=10^9) ---")
ed = all_evaluation_data.get(1e9)
if ed:
    with torch.no_grad():
        outputs_s = model(ed['X_test_tensor']).cpu().numpy()
        outputs_a = y_scaler.inverse_transform(np.clip(outputs_s, 0, 1))
        p_a = outputs_a[:, :5]
        p_skr_direct = outputs_a[:, 5] # NN-Direct log10(SKR)
        
        # Physics-Derived Validation
        p_ks, v_idx = [], []
        for i, (p, l) in enumerate(zip(p_a, ed['fiber_lengths'])):
            k = safe_objective(p, l, 1e9, **PHYSICS_CONSTANTS)
            if k: p_ks.append(k); v_idx.append(i)
        
        print(f"Physics validation: {len(v_idx)}/{len(p_a)} points passed safe_objective check.")
        
        if len(p_a) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Predict on FULL 200km grid for this specific plot
            out_f_s = model(ed['X_test_full_tensor']).cpu().numpy()
            out_f_a = y_scaler.inverse_transform(np.clip(out_f_s, 0, 1))
            p_skr_full = out_f_a[:, 5]
            p_params_full = out_f_a[:, :5]
            
            # Part 1: Key Rate Inheritance (Explicitly showing 150km-200km range)
            ax1.semilogy(ed['fiber_lengths'], np.maximum(ed['optimized_key_rates'], 1e-15), 'b-', label='Hessian Teacher (GT)', linewidth=4)
            # Use FULL grid for student
            ax1.semilogy(ed['full_fiber_lengths'], np.maximum(10**p_skr_full, 1e-15), 'g:', label='NN-Direct (Student)', linewidth=4, alpha=0.8)
            
            # Add validated student rates (where physics allows)
            if len(v_idx) > 0:
                ax1.semilogy(ed['fiber_lengths'][v_idx], np.maximum(p_ks, 1e-15), 'r--', label='Physics-Derived (Student)', linewidth=2)
            
            ax1.set_title(f"Secret Key Rate: 200km Security Cliff ({num_epochs} Epochs)")
            ax1.set_xlabel("Fiber Length (km)"); ax1.set_ylabel("Key Rate (bits/pulse)")
            ax1.set_xlim(0, 200); ax1.set_ylim(1e-15, 1e-2); ax1.grid(True, which='both'); ax1.legend()
            
            # Part 2: Parameter Inheritance
            colors = ['red', 'purple', 'orange', 'green', 'blue']
            labels = ['$\mu_1$', '$\mu_2$', '$P_{\mu_1}$', '$P_{\mu_2}$', '$P_X$']
            for i in range(5):
                # Use FULL grid for parameter plotting
                ax2.plot(ed['full_fiber_lengths'], p_params_full[:, i], color=colors[i], linestyle='--', alpha=0.6)
                ax2.plot(ed['fiber_lengths'], ed['optimized_params_array'][:, i], color=colors[i], label=labels[i])
            ax2.set_title("Parameters Manifold ($n_X=10^9$)")
            ax2.set_xlabel("Fiber Length (km)"); ax2.set_ylabel("Parameter Value")
            ax2.set_ylim(0, 1); ax2.grid(True); ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt_path = os.path.join(output_assets, 'final_performance_diagram.png')
            plt.savefig(plt_path, dpi=300)
            print(f"✓ Final Performance Diagram saved to {plt_path}")
        else:
            print("⚠️ Skipping performance diagram: No points passed physics validation.")

# 6. Training Convergence Plot
print("\n--- Generating Convergence Plot ---")
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.yscale('log')
plt.title('Manifold Inheritance: Training Convergence')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss (Weighted)'); plt.legend(); plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(output_assets, 'training_convergence.png'), dpi=300)
plt.savefig(os.path.join(notebook_dir, 'loss_plot.png'), dpi=300)
print(f"✓ Loss plots saved to {output_assets} and {notebook_dir}")

# 7. Learning Rate Plot
print("\n--- Generating Learning Rate Plot ---")
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, label='Learning Rate', color='green')
plt.yscale('log')
plt.title('Training Learning Rate Schedule')
plt.xlabel('Epoch'); plt.ylabel('LR'); plt.legend(); plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(output_assets, 'learning_rate_plot.png'), dpi=300)
plt.savefig(os.path.join(notebook_dir, 'learning_rate_plot.png'), dpi=300)

print("\n--- Neural Network Training and Analysis Complete ---")
