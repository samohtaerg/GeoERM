# -*- coding: utf-8 -*-
"""# 9 Models Linear Setup

Updated Setup with iteration
"""

from joblib import Parallel, delayed
import numpy as np

def calculate_mse(model, train_data, true_beta, r):
    if model == 'GeoERM':
        result = GeoERM(train_data, r=r+2, link='linear')
        beta_hat = result['step2']
    elif model == 'pERM':
        result = pERM(train_data, r=r, link='linear')
        beta_hat = result['step2'] if isinstance(result, dict) else result
    elif model == 'ERM':
        result = ERM(train_data, r=r, link='linear')
    elif model == 'single_task_LR':
        result = single_task_LR(train_data, link='linear')
    elif model == 'pooled_LR':
        result = pooled_LR(train_data, link='linear')
    elif model == 'spectral':
        result = spectral(train_data, r=r, link='linear')
    elif model == 'MoM':
        result = MoM(train_data, r=r)
    elif model == 'AdaptRep':
        result = AdaptRep(train_data, r=r)
    elif model == 'GLasso':
        result = spectral(train_data, r=r, link='linear')
    else:
        raise ValueError("Unknown model")

    beta_hat = result if isinstance(result, np.ndarray) else result['step2']
    mse = max_distance(beta_hat, true_beta)
    return mse

"""n-task

"""

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main_execution_function(n_tasks=50, n_samples=100, n_features=50, r_list=[3, 5, 10, 15, 20, 25], epsilon=0.1, h=0.1, n_iterations=1):
    models = ['GeoERM', 'pERM', 'ERM', 'single_task_LR', 'pooled_LR', 'spectral', 'MoM', 'AdaptRep', 'GLasso']

    total_results = {r_value: {"no_outliers": {model: [] for model in models},
                               "with_outliers": {model: [] for model in models}}
                     for r_value in r_list}

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}...")

        for r_value in r_list:
            print(f"Running for r = {r_value}...")

            data_no_outliers = generate_data(n=n_samples, p=n_features, r=r_value, T=n_tasks, epsilon=0, h=h, link='linear')
            data_with_outliers = generate_data(n=n_samples, p=n_features, r=r_value, T=n_tasks, epsilon=epsilon, h=h, link='linear')

            mse_results_no_outliers = []
            mse_results_with_outliers = []

            for model in models:
                mse_no_outlier = calculate_mse(model, data_no_outliers['data'], data_no_outliers['beta'], r_value)
                mse_with_outlier = calculate_mse(model, data_with_outliers['data'], data_with_outliers['beta'], r_value)
                mse_results_no_outliers.append(mse_no_outlier)
                mse_results_with_outliers.append(mse_with_outlier)

            for model, mse_no_outlier, mse_with_outlier in zip(models, mse_results_no_outliers, mse_results_with_outliers):
                total_results[r_value]["no_outliers"][model].append(mse_no_outlier)
                total_results[r_value]["with_outliers"][model].append(mse_with_outlier)

    averaged_results = {r_value: {"no_outliers": {model: np.mean(total_results[r_value]["no_outliers"][model]) for model in models},
                                  "with_outliers": {model: np.mean(total_results[r_value]["with_outliers"][model]) for model in models}}
                        for r_value in r_list}

    for r_value in r_list:
        print(f"Averaged MSE results (no outliers, r={r_value}):", averaged_results[r_value]["no_outliers"])
        print(f"Averaged MSE results (with outliers, r={r_value}):", averaged_results[r_value]["with_outliers"])

    import os
    import json
    import sys

    output_dir = ''
    os.makedirs(output_dir, exist_ok=True)

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")  
    averaged_file_path = os.path.join(output_dir, f'averaged_results_{task_id}.json')

    if os.path.exists(averaged_file_path):
        print(f"File {averaged_file_path} exists. Stopping the script.", flush=True)
        sys.exit(0)
    else:
        print(f"File does not exist. Continuing the script.", flush=True)

    with open(averaged_file_path, 'w') as f:
        json.dump(convert_to_serializable(averaged_results), f, indent=4)

    print(f"Results saved to {averaged_file_path}", flush=True)

    return averaged_results

"""# 9 Models Linear Result

"""

# Checklist
# max_iter = 2000 (2000)
# n_samples = 100 (unchanged)
# Task = 50 (50)
# n_features = 50 (50)
#r_list=[5, 10, 15, 20, 25, 30]
#dir
if __name__ == "__main__":
    models = ['GeoERM', 'pERM', 'ERM', 'single_task_LR', 'pooled_LR', 'spectral', 'MoM', 'AdaptRep', 'GLasso']
    logging.basicConfig(level=logging.INFO)

    # Call main_execution_function to get both averaged results and standard errors
    averaged_results = main_execution_function(n_iterations=1)
