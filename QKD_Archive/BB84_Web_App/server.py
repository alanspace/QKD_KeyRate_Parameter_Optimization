from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import joblib

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qkd.model import calculate_key_rates_and_metrics

# Load trained model (Dummy architecture for demo purposes - relies on state_dict)
class BB84Network(nn.Module):
    def __init__(self):
        super(BB84Network, self).__init__()
        # Matches your training architecture: 4 Inputs -> 16 -> 32 -> 16 -> 5 Outputs
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 5)  # Output: mu1, mu2, Pmu1, Pmu2, Px
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

app = Flask(__name__)

# Load Model and Scalers
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../NeuralNetwork/bb84_nn_model.pth')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../NeuralNetwork/models/scaler.pkl')
Y_SCALER_PATH = os.path.join(os.path.dirname(__file__), '../NeuralNetwork/models/y_scaler.pkl')

model = BB84Network()
scaler = None
y_scaler = None

try:
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. Prediction will fail.")

    if os.path.exists(SCALER_PATH) and os.path.exists(Y_SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        print(f"✅ Scalers loaded from NeuralNetwork/models/")
    else:
        print(f"⚠️ Scalers not found. Predictions will be wrong.")
except Exception as e:
    print(f"❌ Error loading resources: {e}")

def prepare_input(fiber_length, n_x):
    # Based on notebook logic:
    # e_1 = fiber_length (km)
    # e_2 = -log10(6e-7) -> fixed P_dc
    # e_3 = 5e-3 * 100 -> fixed e_mis (0.5)
    # e_4 = log10(n_x)
    
    e_1 = float(fiber_length) / 100.0
    e_2 = -np.log10(6e-7) # ~6.22
    e_3 = 5e-3 * 100      # 0.5
    e_4 = np.log10(float(n_x))
    
    # Create array and scale
    raw_input = np.array([[e_1, e_2, e_3, e_4]])
    scaled_input = scaler.transform(raw_input)
    return torch.tensor(scaled_input, dtype=torch.float32)

def prepare_batch_input(inputs_list):
    # inputs_list = [[L, n_x], ...]
    raw_inputs = []
    for item in inputs_list:
        L, nx = item
        e_1 = float(L) / 100.0
        e_2 = -np.log10(6e-7) 
        e_3 = 5e-3 * 100      
        e_4 = np.log10(float(nx))
        raw_inputs.append([e_1, e_2, e_3, e_4])
    
    raw_inputs = np.array(raw_inputs)
    scaled_inputs = scaler.transform(raw_inputs)
    return torch.tensor(scaled_inputs, dtype=torch.float32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        fiber_length = float(data.get('fiber_length', 50))
        n_x = float(data.get('n_x', 1e8))
        
        inputs = prepare_input(fiber_length, n_x)
        
        with torch.no_grad():
            prediction_scaled = model(inputs).numpy() # Scaled output [0-1] roughly
        
        # Inverse transform to get physical values
        prediction = y_scaler.inverse_transform(prediction_scaled)[0]
        # prediction = [mu1, mu2, Pmu1, Pmu2, Px]
        
        mu1, mu2, Pmu1, Pmu2, Px = prediction
        
        return jsonify({
            'status': 'success',
            'results': {
                'mu1': float(mu1),
                'mu2': float(mu2),
                'P_mu1': float(Pmu1),
                'P_mu2': float(Pmu2),
                'P_x': float(Px)
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/plot_data', methods=['POST'])
def plot_data():
    try:
        data = request.get_json()
        n_x = float(data.get('n_x', 1e8))
        
        # 1. Generate Distance Points (0 to 180km) - ensure we cover drop-off
        theory_L = np.linspace(0, 180, 60)
        
        # 2. Batch Predict Optimal Parameters
        inputs_list = [[L, n_x] for L in theory_L]
        tensor_inputs = prepare_batch_input(inputs_list)
        
        with torch.no_grad():
            pk_results_scaled = model(tensor_inputs).numpy()
            
        pk_results = y_scaler.inverse_transform(pk_results_scaled)
        
        # 3. Calculate Exact Key Rate using Physics Engine
        alpha = 0.2
        eta_Bob = 0.1
        P_dc_value = 6e-7
        e_mis = 5e-3
        f_EC = 1.16
        epsilon_sec = 1e-10
        epsilon_cor = 1e-15
        n_event = 1
        P_ap = 0
        
        theory_rates = []
        for i, L in enumerate(theory_L):
            params = pk_results[i] # [mu1, mu2, Pmu1, Pmu2, Px]
            mu_1, mu_2, P_mu_1, P_mu_2, P_X = params
            
            try:
                # Physics call
                results = calculate_key_rates_and_metrics(
                    (mu_1, mu_2, P_mu_1, P_mu_2, P_X), 
                    L, n_x, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
                )
                key_rate = float(results[0])
            except:
                key_rate = 0.0

            if key_rate <= 0 or np.isnan(key_rate):
                key_rate = 1e-15 # Floor for log plot
            
            theory_rates.append(key_rate)
        
        return jsonify({
            'status': 'success',
            'curve_x': theory_L.tolist(),
            'curve_y': theory_rates,
            'point_x': float(data.get('fiber_length', 50))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("🚀 Starting QKD Optimization Server...")
    app.run(host='0.0.0.0', port=8080, debug=True)