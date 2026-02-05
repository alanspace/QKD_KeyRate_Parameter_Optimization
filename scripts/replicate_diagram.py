
import json
import os

def main():
    notebook_path = os.path.join(os.path.dirname(__file__), '../NeuralNetwork/neural_network_updated.ipynb')
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # 1. Update the existing function definition
    function_found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any('def plot_keyrate_and_parameters' in line for line in cell['source']):
            # Found it, let's replace the source
            # We will use a more robust way: find and replace specific lines
            new_source = []
            for line in cell['source']:
                if 'ax_keyrate.set_title(f"Key Rates for $n_X = 5 \\\\times 10^8$")' in line:
                    new_source.append('    exponent = int(np.log10(nx)) if nx and nx > 0 else 0\n')
                    new_source.append('    base = nx / (10**exponent) if nx and nx > 0 else 1\n')
                    new_source.append('    nx_str = f"{base:g} \\\\times 10^{{{exponent}}}" if base != 1 else f"10^{{{exponent}}}"\n')
                    new_source.append('    ax_keyrate.set_title(f"Key Rates for $n_X = {nx_str}$")  # Dynamic title\n')
                elif 'ax_params.set_title(f"Parameters for $n_X = 5 \\\\times 10^8$")' in line:
                    new_source.append('    ax_params.set_title(f"Parameters for $n_X = {nx_str}$")  # Dynamic title\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            function_found = True
            break
    
    if not function_found:
        print("Warning: plot_keyrate_and_parameters not found in notebook.")

    # 2. Add new visualization cell
    new_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 10. Final Visualization: Key Rates and Parameters Overlay\n",
            "Generates the final side-by-side visualization of secret key rates and optimized parameters, comparing the Neural Network predictions against the JAX ground truth."
        ]
    }
    
    new_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Final Visualization Generation\n",
            "nx_vis = 1000000000.0  # Default to 10^9\n",
            "if 'all_evaluation_data' in locals() and nx_vis in all_evaluation_data:\n",
            "    data = all_evaluation_data[nx_vis]\n",
            "    \n",
            "    # Get current model predictions for this data\n",
            "    with torch.no_grad():\n",
            "        X_test_tensor = data['X_test_tensor']\n",
            "        pred_scaled = model(X_test_tensor).cpu().numpy()\n",
            "        pred_scaled = np.clip(pred_scaled, 0, 1)\n",
            "        pred_params = y_scaler.inverse_transform(pred_scaled)\n",
            "        \n",
            "        # Enforce sum constraint on probabilities\n",
            "        prob_sum = pred_params[:, 2] + pred_params[:, 3]\n",
            "        mask_sum_gt_1 = prob_sum > 1.0\n",
            "        if np.any(mask_sum_gt_1):\n",
            "            pred_params[mask_sum_gt_1, 2] /= prob_sum[mask_sum_gt_1]\n",
            "            pred_params[mask_sum_gt_1, 3] /= prob_sum[mask_sum_gt_1]\n",
            "            \n",
            "        # Compute predicted key rates\n",
            "        pred_keys = []\n",
            "        valid_idx = []\n",
            "        for idx, (params, L) in enumerate(zip(pred_params, data['fiber_lengths'])):\n",
            "            key_rate = safe_objective(params, L, nx_vis, alpha=0.2, eta_Bob=0.1, P_dc_value=6e-7, \n",
            "                                      epsilon_sec=1e-10, epsilon_cor=1e-15, f_EC=1.16, \n",
            "                                      e_mis=5e-3, P_ap=0, n_event=1)\n",
            "            if key_rate is not None:\n",
            "                pred_keys.append(key_rate)\n",
            "                valid_idx.append(idx)\n",
            "        \n",
            "        pred_keys = np.array(pred_keys)\n",
            "        \n",
            "        # Generate the diagram\n",
            "        plot_keyrate_and_parameters(\n",
            "            data['fiber_lengths'][valid_idx], \n",
            "            data['optimized_key_rates'][valid_idx], \n",
            "            data['optimized_params_array'][valid_idx],\n",
            "            pred_keys, \n",
            "            pred_params[valid_idx],\n",
            "            epoch='final',\n",
            "            filename=os.path.join(project_root, 'assets/final_performance_diagram.png'),\n",
            "            nx=nx_vis\n",
            "        )\n",
            "else:\n",
            "    print(\"all_evaluation_data not found for 10^9. Please run notebook in order.\")"
        ]
    }
    
    # Check if we already added it (avoid duplicates)
    already_added = any("### 10. Final Visualization" in "".join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'markdown')
    
    if not already_added:
        nb['cells'].append(new_markdown)
        nb['cells'].append(new_code)
        print("Added new visualization cells.")
    else:
        print("Performance visualization cells already exist.")

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully updated {notebook_path}")

if __name__ == "__main__":
    main()
