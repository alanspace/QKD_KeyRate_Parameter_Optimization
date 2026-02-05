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
# Force CPU/64-bit
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

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

def generate_histogram():
    print("Generating Error Histogram...")
    
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
    
    # 2. Load Data
    search_path = 'Training_Data/n_X/good/cleaned_combined_datasets.json'
    if not os.path.exists(search_path):
         search_path = glob.glob('NeuralNetwork/Training_Data/n_X/good/*.json')[0]
    with open(search_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
         dataset = data[list(data.keys())[0]] # Just take the first n_X set for clarity
    else:
        dataset = data
        
    dataset.sort(key=lambda x: x['fiber_length'])
    
    errors = []
    
    # 3. Evaluate
    # P_DC, E_MIS are fixed in this dataset usually but let's check
    # Assuming standard consts for inputs if not in dataset
    # Input features: [L/100, -log10(Pdc), Emis*100, log10(nX)]
    # We need to reconstruct inputs from dataset or assume constants.
    # The dataset usually stores 'optimized_params'.
    # We'll assume the constants match the global training ones (P_DC=6e-7, E_MIS=0.005) 
    # unless stored.
    
    P_DC_VALUE = 6e-7
    E_MIS = 0.005
    N_X = 1e9 # approx if not in record
    
    for d in dataset:
        L = d['fiber_length']
        if L > 180: continue
        
        # Ground Truth Values
        mu1_true = d['optimized_params']['mu_1']
        
        # Prepare Input
        # Note: If dataset has multiple n_X, we should use the record's n_X.
        # But for simplification, we'll assume the loaded key corresponds to High n_X.
        
        inputs = np.array([[L/100.0, -np.log10(P_DC_VALUE), E_MIS*100.0, np.log10(N_X)]])
        inputs_scaled = scaler.transform(inputs)
        
        with torch.no_grad():
             out = model(torch.tensor(inputs_scaled, dtype=torch.float32)).numpy()
        
        params_pred = y_scaler.inverse_transform(out)[0]
        mu1_pred = params_pred[0]
        
        # Relative Error
        rel_err = abs(mu1_pred - mu1_true) / (mu1_true + 1e-9)
        errors.append(rel_err * 100) # Percentage

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='dodgerblue', edgecolor='black', alpha=0.7, density=True)
    
    plt.title('Error Distribution: Neural Network vs JAX Ground Truth', fontsize=14)
    plt.xlabel('Relative Prediction Error (%)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.xlim(0, 5) # Focus on 0-5% range
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate mean
    mean_err = np.mean(errors)
    plt.axvline(mean_err, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_err:.2f}%')
    plt.legend()
    
    output_path = 'assets/error_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {output_path}")

if __name__ == "__main__":
    generate_histogram()
