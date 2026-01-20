
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import glob

def load_latest_dataset():
    project_root = os.path.dirname(os.path.dirname(__file__))
    search_path = os.path.join(project_root, 'Training_Data/n_X/good/reordered_qkd_grouped_dataset_*.json')
    files = sorted(glob.glob(search_path), reverse=True)
    if not files:
        raise FileNotFoundError("No dataset found!")
    with open(files[0], 'r') as f:
        data = json.load(f)
    return data

def main():
    # Use the academic style
    style_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets/paper.mplstyle')
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        print("Warning: paper.mplstyle not found, using defaults.")
    
    data = load_latest_dataset()
    
    # Define n_X values to plot
    # We want to show the progression from 10^4 (unstable/low rate) to 10^9 (asymptotic-like)
    nx_values_to_plot = [100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0]
    
    plt.figure() # Size determined by mplstyle
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(nx_values_to_plot)))
    
    for i, nx in enumerate(nx_values_to_plot):
        nx_key = str(float(nx))
        if nx_key in data:
            entries = data[nx_key]
            # Sort by fiber length
            entries.sort(key=lambda x: x['fiber_length'])
            
            lengths = [e['fiber_length'] for e in entries]
            rates = [e['key_rate'] for e in entries]
            
            # Label formatted for LaTeX (log scale)
            exponent = int(np.log10(nx))
            label = f'$n_X = 10^{{{exponent}}}$'
            
            plt.semilogy(lengths, rates, label=label, color=colors[i], linewidth=2.0)
            
    plt.xlabel(r'Fiber Length ($L$ in km)')
    plt.ylabel(r'Secret Key Rate ($R$ in bps)')
    plt.title(r'Finite-Size Security Analysis: Impact of Block Size ($n_X$)')
    plt.ylim(1e-10, 1e-2)
    plt.xlim(0, 200)
    plt.legend()
    # plt.grid(True, which='both', linestyle='--', alpha=0.3) # handled by style
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets/finite_size_analysis.png')
    plt.savefig(output_path)
    print(f"✅ Generated academic plot: {output_path}")

if __name__ == "__main__":
    main()
