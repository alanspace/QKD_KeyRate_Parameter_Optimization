import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="QKD AI Optimizer",
    page_icon="⚡️",
    layout="wide"
)

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import Physics Engine (Numpy Version for Cloud Compatibility)
try:
    from src.qkd.model_numpy import calculate_key_rates_and_metrics
except ImportError:
    st.error("Could not import logic from `src`. Make sure you are running this from the project root.")
    st.stop()

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class BB84Network(nn.Module):
    def __init__(self):
        super(BB84Network, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 5) # mu1, mu2, Pmu1, Pmu2, Px
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ==========================================
# 3. LOADING RESOURCES (Cached)
# ==========================================
@st.cache_resource
def load_resources():
    # Paths
    MODEL_PATH = os.path.join(current_dir, 'NeuralNetwork/bb84_nn_model.pth')
    SCALER_PATH = os.path.join(current_dir, 'NeuralNetwork/models/scaler.pkl')
    Y_SCALER_PATH = os.path.join(current_dir, 'NeuralNetwork/models/y_scaler.pkl')

    # Load Model
    model = BB84Network()
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
        except:
            st.warning("Model file corrupted or incompatible. Using random weights.")
    else:
        st.warning(f"Model not found at {MODEL_PATH}. Using random weights.")

    # Load Scalers
    try:
        scaler = joblib.load(SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
    except:
        st.error("Scalers not found! Predictions will be inaccurate.")
        return None, None, None

    return model, scaler, y_scaler

model, scaler, y_scaler = load_resources()

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.title("⚡️ QKD Parameter Optimizer")
st.markdown("""
This tool uses a **Neural Network (270x faster than Dual Annealing)** to predict the optimal operating parameters 
for Decoy-State BB84 Quantum Key Distribution.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📡 Experimental Conditions")
    
    fiber_length = st.number_input(
        "Fiber Length (km)", 
        min_value=0.0, 
        max_value=200.0, 
        value=50.0, 
        step=5.0
    )
    
    n_x_exponent = st.slider(
        "Block Size (Log10 n_x)", 
        min_value=5.0, 
        max_value=11.0, 
        value=8.0, 
        step=0.1,
        help="Higher block size means lower finite-size analysis penalties."
    )
    
    n_x = 10**n_x_exponent
    st.caption(f"Actual Block Size: {n_x:,.0f}")
    
    run_btn = st.button("Optimize Parameters 🚀", type="primary")

# ==========================================
# 5. INFERENCE & PLOTTING LOGIC
# ==========================================
if run_btn and model and scaler:
    with st.spinner("Running AI Inference & Physics Verification..."):
        # ---------------------------
        # A. Single Point Prediction
        # ---------------------------
        # Prepare Input: [L/100, -log(Pdc), 0.5, log10(nx)]
        # Scaler expects raw values: [L, 6.22, 0.5, log10(nx)]
        e_1 = float(fiber_length) / 100.0 # Scaling correction we found! 
        e_2 = -np.log10(6e-7) 
        e_3 = 5e-3 * 100      
        e_4 = np.log10(float(n_x))
        
        raw_input = np.array([[e_1, e_2, e_3, e_4]])
        scaled_input = scaler.transform(raw_input)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            pred_scaled = model(input_tensor).numpy()
        
        # Inverse Transform
        pred = y_scaler.inverse_transform(pred_scaled)[0]
        mu1, mu2, Pmu1, Pmu2, Px = pred

        # ---------------------------
        # B. Curve Generation
        # ---------------------------
        # Generate points for plot (0 to 180km)
        curve_L = np.linspace(0, 180, 50)
        
        # Batch Predict Parameters for Curve
        batch_inputs = []
        for L in curve_L:
             batch_inputs.append([float(L)/100.0, e_2, e_3, e_4]) # Correct scaling
             
        batch_raw = np.array(batch_inputs)
        batch_scaled = scaler.transform(batch_raw)
        batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            batch_pred_scaled = model(batch_tensor).numpy()
        
        batch_params = y_scaler.inverse_transform(batch_pred_scaled)
        
        # Calculate Key Rate using Physics Engine
        # Constants
        alpha_val = 0.2
        eta_Bob = 0.1
        P_dc_value = 6e-7
        e_mis = 5e-3
        f_EC = 1.16
        epsilon_sec = 1e-10
        epsilon_cor = 1e-15
        
        key_rates = []
        for i, L_val in enumerate(curve_L):
            p = batch_params[i] # [mu1, mu2, Pmu1, Pmu2, Px]
            try:
                # Physics Call
                res = calculate_key_rates_and_metrics(
                    (p[0], p[1], p[2], p[3], p[4]),
                    L_val, n_x, alpha_val, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, 0, 1
                )
                kr = float(res[0])
                key_rates.append(kr if kr > 0 else 0)
            except:
                key_rates.append(0)

        # Get Key Rate for User Selected Point
        # We can interpolate or calculate exact
        try:
             res_user = calculate_key_rates_and_metrics(
                (mu1, mu2, Pmu1, Pmu2, Px),
                fiber_length, n_x, alpha_val, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, 0, 1
            )
             user_kr = float(res_user[0])
        except:
             user_kr = 0.0

    # ==========================================
    # 6. RESULTS DISPLAY
    # ==========================================
    with col1:
        st.success("Optimization Complete!")
        st.markdown(f"**Max Key Rate:** `{user_kr:.2e}` bits/pulse")
        
        st.markdown("### Optimal Parameters")
        res_df = pd.DataFrame({
            "Parameter": ["Signal (μ1)", "Decoy (μ2)", "Prob (Pμ1)", "Prob (Pμ2)", "Basis (Px)"],
            "Value": [f"{mu1:.3f}", f"{mu2:.3f}", f"{Pmu1:.3f}", f"{Pmu2:.3f}", f"{Px:.3f}"]
        })
        st.table(res_df)

    with col2:
        st.markdown("### Secret Key Rate vs Distance")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Curve
        valid_indices = np.where(np.array(key_rates) > 1e-20)[0]
        if len(valid_indices) > 0:
            ax.semilogy(curve_L[valid_indices], np.array(key_rates)[valid_indices], 
                       label=f"Theoretical Limit (AI Predicted)", linewidth=3, color='#1f77b4', alpha=0.8)
            ax.fill_between(curve_L[valid_indices], np.array(key_rates)[valid_indices], 1e-20, alpha=0.1, color='#1f77b4')
        
        # Plot Point
        if user_kr > 0:
            ax.scatter([fiber_length], [user_kr], color='red', s=150, zorder=5, label="Your Optimized Point")
        
        ax.set_xlabel("Fiber Length (km)", fontsize=12)
        ax.set_ylabel("Secret Key Rate (bits/pulse)", fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.4)
        ax.legend()
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=1e-15, top=1e-2) # Standard QKD range
        
        st.pyplot(fig)

elif model is None:
    st.error("Model failed to load. Please check logs.")
