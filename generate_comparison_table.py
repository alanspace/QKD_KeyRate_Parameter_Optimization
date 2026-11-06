import matplotlib.pyplot as plt
import pandas as pd

# PhD-Grade Presentation Aesthetics
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'figure.dpi': 300,
})

def create_table_image():
    # Transposed Data for Portrait Mode (Methods as Columns)
    # Columns: Metric, Legacy, Neural Surrogate
    data = [
        ["Action Required", "Run Stochastic Global Search", "Single Network Inference"],
        ["Time to Solution", "~1 - 10 Seconds\n(per point)", "1.2 μs\n(per point)"],
        ["Outcome", "Too slow for real-time;\njittery parameters.", "Real-time ready;\nstable, adiabatic parameters."]
    ]
    columns = ["Comparision Metric", "Legacy (Dual Annealing)", "Neural Surrogate"]
    
    # Portrait Layout: Taller figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='left')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 4) # Adjust row height
    
    # Cell formatting
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header Row
            cell.set_text_props(weight='bold', color='white', ha='center')
            cell.set_facecolor('#1f3a5f') # Deep blue header
            cell.set_edgecolor('white')
            cell.set_linewidth(1)
            cell.set_height(0.15)
        else:
            # Data Rows
            cell.set_edgecolor('#dddddd')
            cell.set_linewidth(1)
            cell.set_text_props(ha='left', wrap=True)
            
            # Column 0 (Metric Labels) - styling like a side-header
            if col == 0:
                cell.set_facecolor('#f5f5f5')
                cell.set_text_props(weight='bold')
                cell.set_width(0.25)
            
            # Column 2 (Neural Surrogate) - Highlight
            elif col == 2:
                 cell.set_facecolor('#e8f5e9') # Light green background
            else:
                 cell.set_facecolor('white')

    plt.title("Real-Time Engineering Breakthrough", pad=20, fontsize=20, weight='bold')
    plt.tight_layout()
    
    output_path = 'assets/realtime_engineering_breakthrough.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Table saved to {output_path}")

if __name__ == "__main__":
    create_table_image()
