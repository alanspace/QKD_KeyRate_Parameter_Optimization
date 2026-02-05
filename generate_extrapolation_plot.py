import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from src.qkd.model import calculate_key_rates_and_metrics

# --- Constants ---
NN_BATCH_SIZE = 1000
N_X = 1e9
ALPHA = 0.2
ETA_BOB = 0.1
P_DC_VALUE = 6e-7
EPSILON_SEC = 1e-10
EPSILON_COR = 1e-15
F_EC = 1.16
E_MIS = 0.005
P_AP = 1e-3
N_EVENT = 1

# --- Model Def ---
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

def generate_plot():
    print("Generating Extrapolation Plot...")
    
    # 1. Load Model
    model_path = 'NeuralNetwork/models/bb84_nn_model.pth'
    if not os.path.exists(model_path): model_path = glob.glob('NeuralNetwork/models/*.pth')[0]
    scaler = joblib.load('NeuralNetwork/models/scaler.pkl')
    y_scaler = joblib.load('NeuralNetwork/models/y_scaler.pkl')
    
    model = BB84Network()
    try: model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except: 
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
    model.eval()
    
    # 2. Load Ground Truth (Training Data)
    search_path = 'Training_Data/n_X/good/cleaned_combined_datasets.json'
    if not os.path.exists(search_path):
         search_path = glob.glob('NeuralNetwork/Training_Data/n_X/good/*.json')[0]
    with open(search_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
         dataset = data[list(data.keys())[0]]
    else:
        dataset = data
        
    dataset.sort(key=lambda x: x['fiber_length'])
    
    gt_lengths = []
    gt_rates = []
    for d in dataset:
        l = d['fiber_length']
        if l >= 140: # Zoom in on the tail
            gt_lengths.append(l)
            gt_rates.append(d['key_rate'])
            
    # 3. Generate Extrapolation Curve (140 to 220 km)
    L_eval = np.linspace(140, 220, 100)
    nn_rates = []
    
    for L in L_eval:
        inputs = np.array([[L/100.0, -np.log10(P_DC_VALUE), E_MIS*100.0, np.log10(N_X)]])
        inputs_scaled = scaler.transform(inputs)
        with torch.no_grad():
             out = model(torch.tensor(inputs_scaled, dtype=torch.float32)).numpy()
        
        params_raw = y_scaler.inverse_transform(out)[0]
        p_jax = jnp.array([params_raw[0], params_raw[1], np.clip(params_raw[2],0,1), np.clip(params_raw[3],0,1), np.clip(params_raw[4],0,1)])
        
        skr = calculate_key_rates_and_metrics(
            p_jax, L, N_X, ALPHA, ETA_BOB, P_DC_VALUE, EPSILON_SEC, EPSILON_COR, F_EC, E_MIS, P_AP, N_EVENT
        )[0]
        
        # Physics engine returns 0 or negative for dead zone, we handle small negs
        nn_rates.append(max(float(skr), 1e-15)) # 1e-15 for log plot floor

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Ground Truth
    plt.plot(gt_lengths, gt_rates, 'bo', label='JAX Grond Truth (Training Data)', markersize=5, alpha=0.6)
    
    # Plot NN Curve
    plt.semilogy(L_eval, nn_rates, 'r-', label='Neural Network Prediction', linewidth=2)
    
    # Vertical Line at 180km
    plt.axvline(x=180, color='k', linestyle='--', linewidth=1.5, label='Training Boundary (180km)')
    
    # Region shading
    plt.axvspan(180, 220, color='gray', alpha=0.1, label='Extrapolation Region')
    
    plt.title('Extrapolation Test: "Graceful Decay" Verification', fontsize=14)
    plt.xlabel('Fiber Length (km)', fontsize=12)
    plt.ylabel('Secret Key Rate (bps)', fontsize=12)
    plt.ylim(1e-10, 1e-4)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    output_path = 'assets/extrapolation_check.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_plot()
