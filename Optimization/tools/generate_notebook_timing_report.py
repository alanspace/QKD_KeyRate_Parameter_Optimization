import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob

def generate_report(timing_data, total_sequential, wall_time, num_cores, output_dir):
    """
    Generates a timing summary image and text report based on provided data.
    """
    
    # Sort timing data by n_X
    try:
        nx_labels = sorted(timing_data.keys(), key=lambda x: float(x))
    except:
        nx_labels = list(timing_data.keys())

    times_min = [timing_data[k]/60.0 for k in nx_labels]
    
    # Calculate some stats
    speedup = total_sequential / wall_time if wall_time > 0 else 0
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Bar chart of timing per n_X
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(nx_labels)))
    
    bars = ax1.bar(range(len(nx_labels)), times_min, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('$n_X$ Value', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Optimization Time (minutes)', fontsize=13, fontweight='bold')
    ax1.set_title('Notebook: Optimization Time per $n_X$ Curve', fontsize=14, fontweight='bold')
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
        ['Total Execution Time', f'{total_sequential/60:.2f} min ({total_sequential/3600:.2f} hrs)'],
        ['Wall Time (Parallel)', f'{wall_time/60:.1f} min'],
        ['Estimated Speedup', f'{speedup:.2f}x'],
        ['Number of Cores Used', f'{num_cores}'],
        ['Total n_X Curves', f'{len(timing_data)}'],
        ['Avg Time per n_X', f'{total_sequential/(len(timing_data)):.1f} s'],
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
    
    ax2.set_title('Optimization Performance Summary (Notebook)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'notebook_timing_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Timing summary saved to {output_path}")
    plt.close()
    
    # Also create a simple text report
    report_path = os.path.join(output_dir, 'notebook_timing_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("NOTEBOOK OPTIMIZATION TIMING REPORT\n")
        f.write("=" * 70 + "\n\n")
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
        f.write(f"  Total Execution Time:    {total_sequential:.2f}s ({total_sequential/60:.2f} min / {total_sequential/3600:.2f} hrs)\n")
        f.write(f"  Wall Time (Approximated): {wall_time:.0f}s ({wall_time/60:.1f} min)\n")
        f.write(f"  Speedup Factor:          {speedup:.2f}x\n")
        f.write(f"  Avg Time per n_X:        {total_sequential/(len(timing_data)):.3f}s\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ Text report saved to {report_path}")

def main():
    # Attempt to extract timing data from valid log patterns or simple hardcoded extraction for this specific use case
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "notebook_comparison_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Extracted data from the user provided output stream in the chat
    # Optimizing n_X=1e+04: 100%|██████████| 1000/1000 [02:24<00:00,  6.90it/s]
    # Optimizing n_X=1e+05: 100%|██████████| 1000/1000 [02:41<00:00,  6.20it/s]
    # Optimizing n_X=1e+06: 100%|██████████| 1000/1000 [02:53<00:00,  5.77it/s]
    # Optimizing n_X=1e+07: 100%|██████████| 1000/1000 [02:58<00:00,  5.61it/s]
    # Optimizing n_X=1e+08: 100%|██████████| 1000/1000 [03:00<00:00,  5.54it/s]
    # Optimizing n_X=1e+09: 100%|██████████| 1000/1000 [03:01<00:00,  5.51it/s]
    
    # 2 min 24s = 144s
    # 2 min 41s = 161s
    # 2 min 53s = 173s
    # 2 min 58s = 178s
    # 3 min 00s = 180s
    # 3 min 01s = 181s
    
    timing_data = {
        '1e4': 144.0,
        '1e5': 161.0,
        '1e6': 173.0,
        '1e7': 178.0,
        '1e8': 180.0,
        '1e9': 181.0
    }
    
    # Total Calculation
    total_sequential = sum(timing_data.values())
    
    # Wall time relies on how many cores used. The notebook says "12 threads" (max_workers=6 n_X values)
    # Since there are 6 jobs and 12 cores, they likely all ran continuously in parallel.
    # The wall time is defined by the slowest job (n_X=1e9 took 3:01).
    # Wait, the logs show them interlaced. 
    # Max time is 3:01 = 181s.
    
    wall_time = max(timing_data.values())
    num_cores = 12
    
    generate_report(timing_data, total_sequential, wall_time, num_cores, output_dir)

if __name__ == "__main__":
    main()
