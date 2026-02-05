import matplotlib.pyplot as plt
import numpy as np
import os

def generate_report(timing_data, total_sequential, wall_time, num_cores, output_dir, filename_prefix=""):
    """
    Generates a timing summary image and text report based on provided data.
    """
    
    # Sort timing data by n_X (keys are strings like '1e9', '1e8', etc.)
    # We want 1e4, 1e5... up to 1e9 on the plot
    try:
        nx_labels = sorted(timing_data.keys(), key=lambda x: float(x))
    except:
        nx_labels = list(timing_data.keys())

    times_min = [timing_data[k]/60.0 for k in nx_labels]
    
    # Calculate some stats
    num_points_per_curve = 1000 # detailed run has 1000 points usually
    total_data_points = len(timing_data) * num_points_per_curve # Approximation if not passed explicitly
    speedup = total_sequential / wall_time if wall_time > 0 else 0
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Bar chart of timing per n_X
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(nx_labels)))
    
    bars = ax1.bar(range(len(nx_labels)), times_min, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('$n_X$ Value', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Optimization Time (minutes)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Optimization Time per $n_X$ Curve\n({num_points_per_curve} L-points each)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(nx_labels)))
    
    # Format tick labels
    xtick_labels = []
    for x in nx_labels:
        try:
            val = float(x)
            exponent = int(np.log10(val))
            xtick_labels.append(f'$10^{{{exponent}}}$')
        except:
            xtick_labels.append(x)
    ax1.set_xticklabels(xtick_labels, fontsize=11)
    
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, time_min) in enumerate(zip(bars, times_min)):
        height = bar.get_height()
        nx_key = nx_labels[i]
        seconds = timing_data[nx_key]
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{time_min:.1f} min\n({seconds:.0f}s)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right subplot: Summary table
    ax2.axis('off')
    
    # Create table data
    fastest_nx = min(timing_data, key=timing_data.get)
    slowest_nx = max(timing_data, key=timing_data.get)
    
    table_data = [
        ['Metric', 'Value'],
        ['Total Sequential Time', f'{total_sequential/60:.2f} min ({total_sequential/3600:.2f} hrs)'],
        ['Wall Time (Parallel)', f'{wall_time/60:.1f} min'],
        ['Parallelization Speedup', f'{speedup:.2f}x'],
        ['Number of Cores Used', f'{num_cores}'],
        ['Total n_X Curves', f'{len(timing_data)}'],
        # ['Points per Curve', f'{num_points_per_curve}'],
        # ['Total Data Points', f'{total_data_points}'],
        ['Avg Time per n_X', f'{total_sequential/(len(timing_data)):.1f} s'],
        # ['L Resolution', '0.2 km steps'],
        ['Fastest n_X', f'{fastest_nx} ({timing_data[fastest_nx]/60:.1f} min)'],
        ['Slowest n_X', f'{slowest_nx} ({timing_data[slowest_nx]/60:.1f} min)'],
    ]
    
    # Create table
    table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.45, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            else:
                cell.set_facecolor('white')
            if j == 1:
                cell.set_text_props(weight='bold')
    
    ax2.set_title('Optimization Performance Summary', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{filename_prefix}timing_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Timing summary saved to {output_path}")
    plt.close()
    
    # Also create a simple text report
    report_path = os.path.join(output_dir, f'{filename_prefix}timing_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("QKD OPTIMIZATION TIMING REPORT\n")
        f.write("=" * 70 + "\n\n")
        # f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parallelization: {num_cores} cores\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PER n_X TIMINGS:\n")
        f.write("-" * 70 + "\n")
        for nx in nx_labels:
            time_sec = timing_data[nx]
            f.write(f"  n_X = {nx:>9s}:  {time_sec:7.2f}s  ({time_sec/60:5.2f} min)\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Sequential Time:  {total_sequential:.2f}s ({total_sequential/60:.2f} min / {total_sequential/3600:.2f} hrs)\n")
        f.write(f"  Wall Time (Parallel):   {wall_time:.0f}s ({wall_time/60:.1f} min)\n")
        f.write(f"  Speedup Factor:         {speedup:.2f}x\n")
        f.write(f"  Avg Time per n_X:       {total_sequential/(len(timing_data)):.3f}s\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ Text report saved to {report_path}")

if __name__ == "__main__":
    # Hardcoded data for backward compatibility / testing
    timing_data = {
        '1e9': 1372.84,
        '1e8': 1397.56,
        '1e7': 1425.85,
        '1e6': 1487.99,
        '1e5': 1612.57,
        '1e4': 1922.34
    }
    
    total_sequential = 9219.17  # seconds
    wall_time = 32 * 60  # ~32 minutes
    num_cores = 12
    # output_dir = '/Users/daai6ga1hou2/Documents/GitHub/Physics/QKD_KeyRate_Parameter_Optimization/Optimization/script_version/results_global_jax/'
    output_dir = os.path.dirname(os.path.abspath(__file__)) + "/results_global_jax"
    
    generate_report(timing_data, total_sequential, wall_time, num_cores, output_dir)

