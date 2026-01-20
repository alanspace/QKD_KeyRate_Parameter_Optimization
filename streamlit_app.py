import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
import sys
import time

# FORCE JAX to CPU (Metal has issues with JIT-ted float64/complex)
os.environ['JAX_PLATFORMS'] = 'cpu'

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

# ==========================================
# 2. HYBRID PHYSICS ENGINE LOADING
# ==========================================
# Try loading JAX (Fast/GPU) first, fall back to Numpy (Cloud/CPU)
engine_type = "Unknown"
try:
    import jax
    from src.qkd.model import calculate_key_rates_and_metrics
    engine_type = "🚀 JAX (High Performance)"
    print("Success: Loaded JAX engine.")
except ImportError:
    from src.qkd.model_numpy import calculate_key_rates_and_metrics
    engine_type = "☁️ Numpy (Cloud Compatibility)"
    print("Fallback: Loaded Numpy engine.")
except Exception as e:
    st.error(f"Critical Error loading physics engine: {e}")
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
# ==========================================
# 3. LOADING RESOURCES (Cached)
# ==========================================
@st.cache_resource
def load_scalers():
    SCALER_PATH = os.path.join(current_dir, 'NeuralNetwork/models/scaler.pkl')
    Y_SCALER_PATH = os.path.join(current_dir, 'NeuralNetwork/models/y_scaler.pkl')
    try:
        scaler = joblib.load(SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        return scaler, y_scaler
    except Exception as e:
        st.error(f"Scalers load error: {e}")
        return None, None

@st.cache_resource
def load_model(model_type):
    if model_type == "Modern (JAX)":
        path = os.path.join(current_dir, 'NeuralNetwork/models/bb84_nn_model_jax.pth')
    else:
        path = os.path.join(current_dir, 'NeuralNetwork/models/bb84_nn_model_legacy.pth')
    
    model = BB84Network()
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            st.warning(f"Error loading {model_type}: {e}")
    return None

# Load Scalers once
scaler, y_scaler = load_scalers()

# Sidebar Selector
st.sidebar.header("🧠 AI Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model Version",
    ["Modern (JAX)", "Legacy (Annealing)"],
    help="Modern: Trained on JAX data (High Accuracy)\nLegacy: Trained on Annealing data (Noisier)"
)

# Load Selected Model
model = load_model(model_choice)

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.title("⚡️ QKD Parameter Optimizer")
st.markdown(f"""
**Using Model:** `{model_choice}`
Running on: **{engine_type}**
""")

tab1, tab2 = st.tabs(["🚀 Live Optimizer", "📊 System Analysis"])

with tab1:
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
        start_time = time.time()
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
                    key_rates.append(kr if kr > 1e-20 else 0)
                except Exception as e:
                    # Log error to console for debugging
                    print(f"Physics error at {L_val}km: {e}")
                    key_rates.append(0)

            # Get Key Rate for User Selected Point
            # We can interpolate or calculate exact
            try:
                 res_user = calculate_key_rates_and_metrics(
                    (mu1, mu2, Pmu1, Pmu2, Px),
                    fiber_length, n_x, alpha_val, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, 0, 1
                )
                 user_kr = float(res_user[0])
            except Exception as e:
                 print(f"User point physics error: {e}")
                 user_kr = 0.0

        # ==========================================
        # 6. RESULTS DISPLAY
        # ==========================================
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        with col1:
            st.success("Optimization Complete!")
            st.info(f"⚡️ Processing Time: {elapsed_time:.4f} seconds")
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
            ax.set_ylim(bottom=1e-7, top=1e-2) # Adjusted QKD range
            
            st.pyplot(fig)

with tab2:
    st.header("📊 Deep System Analysis")
    st.markdown("""
    This section provides the scientific verification of the AI's performance and the underlying physics of the optimization.
    """)
    
    # 1. Performance Metrics
    st.subheader("🏁 Verified Performance Gains")
    m1, m2, m3 = st.columns(3)
    m1.metric("Range Extension", "+5.0 km", help="Max distance increase vs. Static Parameters")
    m2.metric("Peak Rate Gain", "69x", help="Key rate improvement at 180km")
    m3.metric("AI Speedup", ">50,000x", help="Neural Network vs. traditional Global Optimization")
    
    # 2. Visual Proofs
    cola, colb = st.columns(2)
    
    with cola:
        st.markdown("#### 🎯 Parameter Sensitivity")
        st.caption("Which parameters affect the Key Rate most?")
        if os.path.exists("Testing/Sensitivity_Analysis.png"):
            st.image("Testing/Sensitivity_Analysis.png", use_container_width=True)
            st.info("**Insight:** Basis Probability ($P_X$) is the most critical factor. A 20% error leads to 100% signal loss, proving why precision optimization is vital.")
        else:
            st.warning("Sensitivity plot not found. Run `Analysis/parameter_sensitivity.py` to generate.")

    with colb:
        st.markdown("#### ⚡️ Dynamic vs. Static")
        st.caption("Comparison of AI-optimized vs. Fixed parameters.")
        if os.path.exists("Testing/Dynamic_vs_Static_Overlay.png"):
            st.image("Testing/Dynamic_vs_Static_Overlay.png", use_container_width=True)
        else:
            st.warning("Comparison plot not found.")

    st.divider()
    st.markdown("### 🧬 Professional Audit Trail")
    st.markdown("""
    - **Optimization Strategy**: JAX-Accelerated Multi-Start Hybrid Search.
    - **Model Architecture**: 4-Layer Feed-Forward Neural Network (PyTorch).
    - **Training State**: 5,000 Epochs with Learning Rate Decay.
    - **Physics Core**: Finite-Key Decoy-State BB84 (Lim et al., 2014).
    """)

if model is None:
    st.error("Model failed to load. Please check logs.")
