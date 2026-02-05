import os
import shutil
import glob
from pathlib import Path

def organize_project():
    # 1. Define the Professional Structure
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    
    config = {
        "results/plots/latest": [
            "Testing/Dynamic_vs_Static_Overlay.png",
            "Testing/Sensitivity_Analysis.png",
            "NeuralNetwork/parameter_relative_error_nx_1e+09.png", # Moving current best
        ],
        "results/plots/training/6000_epochs": ["NeuralNetwork/6000_epochs/*.png"],
        "results/plots/training/5000_epochs": [
            "NeuralNetwork/5000_epochs/16_32/*.png",
            "NeuralNetwork/5000_epochs/40_80/*.png",
            "NeuralNetwork/5000_epochs/50_100/*.png"
        ],
        "results/plots/archive/legacy_plots": [
            "NeuralNetwork/parameter_relative_error_nx_*.png",
            "NeuralNetwork/learning_rate_plot.png",
            "NeuralNetwork/loss_plot.png",
            "NeuralNetwork/parameters_subplots_*.png",
            "NeuralNetwork/keyrate_subplots_*.png",
            "NeuralNetwork/image/*.png",
            "NeuralNetwork/image/legacy/*.png",
            "NeuralNetwork/image/legacy/*.jpeg"
        ],
        "results/plots/archive/jax_previews": [
            "NeuralNetwork/Training_Data/n_X/good/*.png",
            "NeuralNetwork/n_X/good/*.png"
        ]
    }

    print(f"--- Starting Pro Reorganization in {PROJECT_ROOT} ---")
    
    # 2. execute Moves
    for target_subpath, patterns in config.items():
        target_dir = os.path.join(PROJECT_ROOT, target_subpath)
        os.makedirs(target_dir, exist_ok=True)
        
        for pattern in patterns:
            # Handle absolute paths vs relative globbing
            full_pattern = os.path.join(PROJECT_ROOT, pattern)
            found_files = glob.glob(full_pattern)
            
            for file_path in found_files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(target_dir, filename)
                
                # Check if we are moving a file to itself (can happen if running multiple times)
                if os.path.abspath(file_path) == os.path.abspath(dest_path):
                    continue

                try:
                    shutil.move(file_path, dest_path)
                    print(f"Moved: {os.path.basename(file_path)} -> {target_subpath}/")
                except Exception as e:
                    print(f"Skipped {filename}: {e}")

    # 3. Clean up empty directories
    dirs_to_prune = [
        "NeuralNetwork/image/legacy",
        "NeuralNetwork/image",
        "NeuralNetwork/6000_epochs",
        "NeuralNetwork/5000_epochs",
        "Testing"
    ]
    
    for dir_rel in dirs_to_prune:
        dir_full = os.path.join(PROJECT_ROOT, dir_rel)
        try:
            if os.path.exists(dir_full) and not os.listdir(dir_full):
                os.rmdir(dir_full)
                print(f"Removed empty dir: {dir_rel}")
        except Exception as e:
            pass 

    print("--- Reorganization Complete ---")

if __name__ == "__main__":
    organize_project()
