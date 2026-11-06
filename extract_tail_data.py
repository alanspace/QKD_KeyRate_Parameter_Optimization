import json
import os

def extract_tail_range(json_path, nx_key, min_L, max_L):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if nx_key not in data:
        return []
        
    entries = data[nx_key]
    tail_data = []
    
    for entry in entries:
        r = entry.get('key_rate', 0)
        l = entry.get('fiber_length', 0)
        
        if min_L <= l <= max_L and r > 1e-25:
            params = entry.get('optimized_params', {})
            tail_data.append({
                'L': l,
                'Key Rate': r,
                'mu_1': params.get('mu_1'),
                'mu_2': params.get('mu_2'),
                'P_mu_1': params.get('P_mu_1'),
                'P_mu_2': params.get('P_mu_2'),
                'P_X': params.get('P_X_value')
            })
            
    return tail_data

def list_of_dicts_to_md(data):
    if not data:
        return "No valid data points found in this range.\n"
    
    headers = list(data[0].keys())
    md = "|" + "|".join(headers) + "|\n"
    md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    
    for row in data:
        row_str = "|"
        for key in headers:
            val = row[key]
            if isinstance(val, float):
                if val < 1e-5 or val > 1e4:
                    row_str += f"{val:.4e}|"
                else:
                    row_str += f"{val:.6f}|"
            else:
                 row_str += f"{val}|"
        md += row_str + "\n"
    return md

def main():
    hessian_path = '/Users/daai6ga1hou2/Documents/GitHub/Physics/QKD_KeyRate_Parameter_Optimization/Optimization/run_scripts/outputs/JAX/Hessian_FoR/run_HESSIAN_FoR_20260217_014536/qkd_grouped_dataset_20260217_014536.json'
    da_path = '/Users/daai6ga1hou2/Documents/GitHub/Physics/QKD_KeyRate_Parameter_Optimization/Optimization/run_scripts/outputs/JAX/Pure_DA/run_PURE_DA_20260216_003022/qkd_grouped_dataset_20260216_003022.json'
    
    output_md = '/Users/daai6ga1hou2/Documents/GitHub/Physics/QKD_KeyRate_Parameter_Optimization/assets/tail_region_analysis_140_150km.md'
    
    hessian_data = extract_tail_range(hessian_path, '10000.0', 140, 150)
    da_data = extract_tail_range(da_path, '10000.0', 140, 150)
    
    with open(output_md, 'w') as f:
        f.write("# Tail Region Numerical Analysis (140km - 150km)\n")
        f.write("Target: $n_X = 10^4$\n\n")
        
        f.write("## Forward-over-Reverse Hessian (Trust-NCG)\n")
        f.write(list_of_dicts_to_md(hessian_data))
            
        f.write("\n\n## Pure Dual Annealing\n")
        f.write(list_of_dicts_to_md(da_data))
            
    print(f"Data extracted to {output_md}")

if __name__ == '__main__':
    main()
