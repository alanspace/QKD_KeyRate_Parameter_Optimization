# Import necessary libraries
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

# Set default tensor type to float64 for higher precision
torch.set_default_dtype(torch.float64)

# Helper functions for tensor operations
def to_tensor(x):
    """Convert input to PyTorch tensor and move to device"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)

def calculate_factorial(n):
    """Calculate factorial using torch.lgamma"""
    return torch.exp(torch.lgamma(n + 1))

def calculate_tau_n(n, mu):
    """Calculate tau_n using PyTorch operations"""
    mu = to_tensor(mu)
    n = to_tensor(n)
    return (mu**n * torch.exp(-mu)) / calculate_factorial(n)

def calculate_eta_ch(L, alpha):
    """Calculate channel transmittance"""
    L = to_tensor(L)
    alpha = to_tensor(alpha)
    return torch.exp(-alpha * L / 10)

def calculate_eta_sys(eta_ch, eta_Bob):
    """Calculate system transmittance"""
    return eta_ch * eta_Bob

class QKDOptimizer(nn.Module):
    """PyTorch module for QKD parameter optimization"""
    def __init__(self):
        super().__init__()
        # Initialize parameters as nn.Parameter for automatic optimization
        self.mu_1 = nn.Parameter(torch.tensor(0.6))
        self.mu_2 = nn.Parameter(torch.tensor(0.2))
        self.P_mu_1 = nn.Parameter(torch.tensor(0.65))
        self.P_mu_2 = nn.Parameter(torch.tensor(0.3))
        self.P_X = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, L, n_X, alpha, eta_Bob, P_dc, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
        """Calculate key rate for given parameters"""
        # Ensure parameters are within bounds using sigmoid
        P_mu_1 = torch.sigmoid(self.P_mu_1)
        P_mu_2 = torch.sigmoid(self.P_mu_2) * (1 - P_mu_1)
        P_X = torch.sigmoid(self.P_X)
        
        # Calculate transmittance
        eta_ch = calculate_eta_ch(L, alpha)
        eta_sys = calculate_eta_sys(eta_ch, eta_Bob)
        
        # Calculate key rate components
        # ... (implement key rate calculation using PyTorch operations)
        
        return key_rate

def optimize_parameters(L, n_X, alpha, eta_Bob, P_dc, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """Optimize QKD parameters for given conditions"""
    model = QKDOptimizer().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    best_key_rate = float('-inf')
    best_params = None
    
    for epoch in range(1000):
        optimizer.zero_grad()
        key_rate = model(L, n_X, alpha, eta_Bob, P_dc, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
        loss = -key_rate  # Maximize key rate by minimizing negative
        loss.backward()
        optimizer.step()
        
        if key_rate.item() > best_key_rate:
            best_key_rate = key_rate.item()
            best_params = {
                'mu_1': model.mu_1.item(),
                'mu_2': model.mu_2.item(),
                'P_mu_1': torch.sigmoid(model.P_mu_1).item(),
                'P_mu_2': torch.sigmoid(model.P_mu_2).item() * (1 - torch.sigmoid(model.P_mu_1).item()),
                'P_X': torch.sigmoid(model.P_X).item()
            }
    
    return best_key_rate, best_params

def generate_dataset_by_n_X(Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """
    Generate dataset organized by n_X values
    """
    # Initialize dictionary to store results by n_X
    categorized_dataset = {float(n_X): [] for n_X in n_X_values}  # Convert to float for dictionary keys
    
    print("Generating dataset...")
    with tqdm(total=len(Ls) * len(n_X_values), desc="Generating Dataset") as progress_bar:
        for L in Ls:
            for n_X in n_X_values:
                # Convert JAX array to float for the optimization
                n_X_float = float(n_X)
                
                # Optimize parameters for this combination
                result = optimize_single_instance(
                    (L, n_X_float), bounds, alpha, eta_Bob, P_dc_value, 
                    epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
                )
                
                # Unpack results
                L_val, _, penalized_key_rate, optimized_params = result
                
                # Skip invalid key rates
                if penalized_key_rate <= 0:
                    progress_bar.update(1)
                    continue
                
                # Extract optimized parameters
                mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = optimized_params
                
                # Compute normalized parameters
                e_1 = float(L_val / 100)  # Normalize fiber length
                e_2 = float(-jnp.log10(P_dc_value))  # Normalize dark count probability
                e_3 = float(e_mis * 100)  # Normalize misalignment error probability
                e_4 = float(jnp.log10(n_X_float))  # Normalize number of pulses
                
                # Append processed data into the categorized dictionary
                categorized_dataset[n_X_float].append({
                    "fiber_length": float(L_val),
                    "e_1": e_1,
                    "e_2": e_2,
                    "e_3": e_3,
                    "e_4": e_4,
                    "key_rate": float(max(penalized_key_rate, 1e-10)),
                    "optimized_params": {
                        "mu_1": float(mu_1),
                        "mu_2": float(mu_2),
                        "P_mu_1": float(P_mu_1),
                        "P_mu_2": float(P_mu_2),
                        "P_X_value": float(P_X_value)
                    }
                })
                
                progress_bar.update(1)
    
    # Save the dataset
    with open('training_dataset_by_nx.json', 'w') as f:
        json.dump(categorized_dataset, f, indent=2)
    
    return categorized_dataset

def plot_grouped_parameters(dataset, n_X_values):
    """
    Plot grouped parameters and key rates for multiple n_X values
    """
    # Create two subplots: one for key rates, one for other parameters
    fig, (ax_key_rate, ax_params) = plt.subplots(2, 1, figsize=(15, 20))
    
    # Plot key rates for each n_X value
    for n_X in n_X_values:
        if n_X not in dataset:
            print(f"No data found for n_X = {n_X}")
            continue
            
        data = dataset[n_X]
        data.sort(key=lambda x: x['fiber_length'])
        
        fiber_lengths = [d['fiber_length'] for d in data]
        key_rates = [d['key_rate'] for d in data]
        
        # Plot key rate
        ax_key_rate.plot(fiber_lengths, key_rates, label=f'n_X = {n_X:.0e}')
    
    # Configure key rate plot
    ax_key_rate.set_xlabel('Fiber Length (km)')
    ax_key_rate.set_ylabel('Key Rate')
    ax_key_rate.set_yscale('log')
    ax_key_rate.grid(True)
    ax_key_rate.legend()
    ax_key_rate.set_title('Key Rate vs Fiber Length')
    
    # Plot other parameters (using first n_X value for parameters)
    n_X = n_X_values[0]
    data = dataset[n_X]
    data.sort(key=lambda x: x['fiber_length'])
    
    fiber_lengths = [d['fiber_length'] for d in data]
    params = {
        'μ₁': [d['optimized_params']['mu_1'] for d in data],
        'μ₂': [d['optimized_params']['mu_2'] for d in data],
        'P_μ₁': [d['optimized_params']['P_mu_1'] for d in data],
        'P_μ₂': [d['optimized_params']['P_mu_2'] for d in data],
        'P_X': [d['optimized_params']['P_X_value'] for d in data]
    }
    
    # Plot each parameter
    for param_name, values in params.items():
        ax_params.plot(fiber_lengths, values, label=param_name)
    
    # Configure parameters plot
    ax_params.set_xlabel('Fiber Length (km)')
    ax_params.set_ylabel('Parameter Value')
    ax_params.set_yscale('log')
    ax_params.grid(True)
    ax_params.legend()
    ax_params.set_title('Optimized Parameters vs Fiber Length')
    
    plt.tight_layout()
    plt.show()

# Usage example:
if __name__ == "__main__":
    # Generate dataset for a smaller range first
    Ls = jnp.linspace(5, 200, 100)  # Fewer points for testing
    n_X_values = [1e6, 1e7]  # Just two values for testing
    
    # Generate the dataset
    dataset = generate_dataset_by_n_X(
        Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, 
        epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
    )
    
    # Plot grouped parameters and key rates
    plot_grouped_parameters(dataset, n_X_values) 