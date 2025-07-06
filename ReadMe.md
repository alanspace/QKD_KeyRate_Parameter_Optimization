To Read the report, please click

https://www.overleaf.com/read/gdxppttgfrbb#c1cf24

To read all the functions, please click

https://gits-15.sys.kth.se/slleung/QKD_KeyRate_ParameterOptimization/blob/170010d2cd376acb9b0b948064303ab7deb2dc2a/QKD_Functions/QKD_Functions.py


To read the notebook on simulation, please click 

https://gits-15.sys.kth.se/slleung/QKD_KeyRate_ParameterOptimization/blob/main/Analysis/BB84_Parameters_2014_Analysis_Jax.ipynb

To read the notebook on optimization, please click

https://gits-15.sys.kth.se/slleung/QKD_KeyRate_ParameterOptimization/blob/main/Optimization/BB84_Parameters_2014_Optimization_Jax.ipynb

To read the notebook on neural network, please click

https://gits-15.sys.kth.se/slleung/QKD_KeyRate_ParameterOptimization/blob/main/NeuralNetwork/nerualnetwork.ipynb


Metrics Calculation:
	•	The call to calculate_key_rates_and_metrics ensures that all required metrics are computed in one step.
	•	The computation of penalties is encapsulated in the penalty function.

Return Values:
	•	The function returns both the penalized key rates and all other metrics.

Optimization
•	The optimization implicitly affects the other parameters (mu_1, mu_2, P_mu_1, P_mu_2, P_X). 
•	The optimization algorithm adjusts these parameters to find the configuration that maximizes the penalized key rate. 
•	These parameters are the decision variables that are tuned during optimization, and the final optimized values are extracted as optimized_params in the code.

Global Optimization (Dual Annealing):
	•	The use of dual_annealing is for a global optimizer as it explores the parameter space broadly.
	•	The function wrapped_objective ensures that objective is compatible with the optimization interface.

Local Optimization (Nelder-Mead):
	•	Refining the result of dual_annealing with Nelder-Mead ensures fine-tuning around the global minimum found by the first step.
	•	x0=global_result.x is used to initialize Nelder-Mead.

Result Handling:
	•	optimized_params captures the fine-tuned parameters, and optimized_key_rate ensures the key rate is converted back to positive.

In the dataset generation, there are data being storing as follow:
	•	Normalized parameters (e_1, e_2, e_3, e_4).
	•	Penalized key rates.
	•	Optimized parameters (mu_1, mu_2, P_mu_1, P_mu_2, P_X_value).

leveraging jax.experimental.maps for a more optimized approach 

# Reference:
1. Wang, W., & Lo, H. K. (2019). Machine learning for optimal parameter prediction in quantum key distribution. Physical Review A, 100(6), 062334
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.062334
    
2. Lim, C. C. W., Curty, M., Walenta, N., Xu, F., & Zbinden, H. (2014). Concise security bounds for practical decoy-state quantum key distribution. Physical Review A, 89(2), 022307.
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.89.022307
