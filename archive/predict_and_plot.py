import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from QKD_Functions.QKD_Functions import objective, calculate_key_rates_and_metrics

class BB84NN(torch.nn.Module):
    def __init__(self):
        super(BB84NN, self).__init__()
        self.fc1 = torch.nn.Linear(4, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 5)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

def prepare_input(L, n_X):
    """Prepare normalized input features for the model"""
    e_1 = L / 100  # Normalize fiber length
    e_2 = -np.log10(6e-7)  # Dark count probability
    e_3 = 5e-3 * 100  # Misalignment error
    e_4 = np.log10(n_X)  # Number of pulses
    
    return np.array([e_1, e_2, e_3, e_4], dtype=np.float32)

def calculate_and_plot():
    # Check if model exists
    model_path = "../NeuralNetwork/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model
    model = BB84NN()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # System parameters
    alpha = 0.2
    eta_Bob = 0.1
    P_dc_value = 6e-7
    epsilon_sec = 1e-10
    epsilon_cor = 1e-15
    f_EC = 1.16
    e_mis = 5e-3
    P_ap = 0
    n_event = 1
    n_X = 1e12

    # Generate fiber lengths
    L_values = np.linspace(0, 200, 1000)
    key_rates = []
    predicted_params = []
    
    print("Calculating key rates...")
    for L in L_values:
        # Prepare input for the model
        X = prepare_input(L, n_X)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        
        # Get predictions from the model
        with torch.no_grad():
            params = model(X_tensor).numpy()[0]
        predicted_params.append(params)
        
        # Calculate metrics using the predicted parameters
        results = calculate_key_rates_and_metrics(
            params, L, n_X, alpha, eta_Bob, P_dc_value, 
            epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        key_rate = results[0]  # First return value is key_rate
        key_rates.append(float(key_rate))
    
    # Convert to numpy arrays
    key_rates = np.array(key_rates)
    predicted_params = np.array(predicted_params)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(L_values, key_rates, 'b-', label=f'n_X = {n_X:.0e}')
    plt.xlabel('Fiber Length (km)')
    plt.ylabel('Key Rate')
    plt.yscale('log')
    plt.title('Predicted Key Rate vs Fiber Length')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    param_names = ['μ₁', 'μ₂', 'P_μ₁', 'P_μ₂', 'P_X']
    for i in range(5):
        plt.plot(L_values, predicted_params[:, i], label=param_names[i])
    plt.xlabel('Fiber Length (km)')
    plt.ylabel('Parameter Value')
    plt.title('Predicted Parameters vs Fiber Length')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'fiber_lengths': L_values.tolist(),
        'key_rates': key_rates.tolist(),
        'predicted_parameters': predicted_params.tolist(),
        'n_X': float(n_X),
        'parameters': {
            'alpha': alpha,
            'eta_Bob': eta_Bob,
            'P_dc': P_dc_value,
            'epsilon_sec': epsilon_sec,
            'epsilon_cor': epsilon_cor,
            'f_EC': f_EC,
            'e_mis': e_mis
        }
    }
    
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Results saved to predictions.json")

if __name__ == "__main__":
    calculate_and_plot() 