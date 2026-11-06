import matplotlib.pyplot as plt
import numpy as np
import os

# PhD-Grade Presentation Aesthetics
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.dpi': 300,
})

def plot_comparison():
    # Data extracted from timing reports
    # Metric: Average Time per n_X Regime (Curve Generation)
    
    # 1. Pure Dual Annealing (JAX)
    # Source: Optimization/run_scripts/outputs/JAX/Pure_DA/run_PURE_DA_20260216_003022/timing_report.txt
    # Avg Time per n_X: 708.981s
    t_pure_da = 708.98
    
    # 2. Hessian Forward-over-Reverse (JAX)
    # Source: Optimization/run_scripts/outputs/JAX/Hessian_FoR/run_HESSIAN_FoR_20260217_014536/timing_report.txt
    # Avg Time per n_X: 78.888s
    t_hessian = 78.89
    
    # 3. Neural Surrogate
    # Source: benchmark_inference.py result (0.0012 ms per point)
    # Assumption: 100 points per regime to match the optimization curves
    t_neural = (0.0012 / 1000) * 100 # seconds per regime
    
    methods = ['Pure Dual Annealing\n(Global Search)', 'Hessian FoR\n(Physics-Aware)', 'Neural Surrogate\n(AI Prediction)']
    times = [t_pure_da, t_hessian, t_neural]
    colors = ['#5c6bc0', '#26a69a', '#ef5350'] # Indigo, Teal, Red
    
    plt.figure(figsize=(12, 7))
    bars = plt.barh(methods, times, color=colors)
    
    plt.xscale('log')
    plt.xlabel('Time to Generate One Security Regime (Seconds) - Log Scale')
    plt.title('Optimization Method Speed Comparison')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis() # Top to bottom: Slowest to Fastest
    
    # Annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if i == 0:
            label = f"{width:.1f} s\n(~12 min)"
            plt.text(width * 1.1, bar.get_y() + bar.get_height()/2, label, va='center', fontweight='bold', color=colors[i])
        elif i == 1:
            speedup = t_pure_da / width
            label = f"{width:.1f} s\n({int(speedup)}x Faster)"
            plt.text(width * 1.1, bar.get_y() + bar.get_height()/2, label, va='center', fontweight='bold', color=colors[i])
        elif i == 2:
            speedup_vs_hessian = t_hessian / width
            speedup_vs_da = t_pure_da / width
            ms = width * 1000
            label = f"{ms:.2f} ms\n(>{int(speedup_vs_hessian):,}x vs Hessian)\n(>{int(speedup_vs_da):,}x vs DA)"
            plt.text(width * 1.5, bar.get_y() + bar.get_height()/2, label, va='center', fontweight='bold', color=colors[i])
            
    # Limits
    plt.xlim(1e-5, 2000)
    plt.tight_layout()
    
    output_path = 'assets/optimization_speed_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    plot_comparison()
