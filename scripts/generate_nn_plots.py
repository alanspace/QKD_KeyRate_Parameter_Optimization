
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os
import joblib

# Define the model class (must match the one used for training)
class BB84Network(nn.Module):
    def __init__(self):
        super(BB84Network, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_data_and_model():
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root, 'NeuralNetwork/models/bb84_nn_model_jax.pth')
    scaler_path = os.path.join(project_root, 'NeuralNetwork/models/scaler.pkl')
    y_scaler_path = os.path.join(project_root, 'NeuralNetwork/models/y_scaler.pkl')
    
    # Load dataset
    search_path = os.path.join(project_root, 'Training_Data/n_X/good/reordered_qkd_grouped_dataset_*.json')
    files = sorted(glob.glob(search_path), reverse=True)
    if not files:
        raise FileNotFoundError("No dataset found.")
    
    with open(files[0], 'r') as f:
        data = json.load(f)
        
    # Load scalers
    scaler = joblib.load(scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    
    # Load model
    model = BB84Network()
    # Check if MPS is available (Mac)
    device = torch.device("cpu") # specific to cpu for plotting consistency
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return data, model, scaler, y_scaler

def generate_error_plot():
    data, model, scaler, y_scaler = load_data_and_model()
    
    # Target n_X
    nx_target = 1000000000.0 # 10^9
    nx_key = str(nx_target)
    
    if nx_key not in data:
        print(f"n_X={nx_target} not found in dataset.")
        return

    entries = data[nx_key]
    entries.sort(key=lambda x: x['fiber_length'])
    
    fiber_lengths = np.array([e['fiber_length'] for e in entries])
    
    # Prepare Input X
    # Input features: e_1 (fiber_length/100), e_2, e_3, e_4 (log10(n_X))
    # We reconstruct them as per the training logic
    X_input = []
    ground_truth_params = []
    
    for e in entries:
        e_1 = e['fiber_length'] / 100.0
        e_2 = -np.log10(6e-7) # Fixed P_dc
        e_3 = 5e-3 * 100 # Fixed e_mis
        e_4 = np.log10(nx_target)
        X_input.append([e_1, e_2, e_3, e_4])
        
        # Ground truth parameters
        p = e['optimized_params']
        ground_truth_params.append([p['mu_1'], p['mu_2'], p['P_mu_1'], p['P_mu_2'], p['P_X_value']])

    X_input = np.array(X_input)
    ground_truth_params = np.array(ground_truth_params)
    
    # Scale Input
    X_scaled = scaler.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        predictions_scaled = model(X_tensor).numpy()
        
    # Inverse Scale Predictions
    predictions = y_scaler.inverse_transform(predictions_scaled)
    
    # Calculate Relative Error
    # Avoid division by zero
    denominator = np.maximum(ground_truth_params, 1e-10)
    relative_errors = (predictions - ground_truth_params) / denominator
    
    # Plotting
    # Parameters: mu_1, mu_2, P_mu_1, P_mu_2, P_X
    param_labels = [r'$\mu_1$', r'$\mu_2$', r'$P_{\mu_1}$', r'$P_{\mu_2}$', r'$P_X$']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(5):
        ax = axes[i]
        ax.plot(fiber_lengths, relative_errors[:, i], linewidth=2, label='Relative Error')
        ax.set_title(f'Relative Error: {param_labels[i]}')
        ax.set_xlabel('Fiber Length (km)')
        ax.set_ylabel('Rel. Error (Pred - True)/True')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-0.05, 0.05) # Zoom in to see the <1% error clearly
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets/parameter_relative_error_nx_1e09.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ Error plot saved to {output_path}")

if __name__ == "__main__":
    generate_error_plot()
