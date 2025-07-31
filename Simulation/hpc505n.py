# -*- coding: utf-8 -*-
"""# 9 Models Linear Setup

Updated Setup with iteration
"""

from joblib import Parallel, delayed
import numpy as np


# Helper function to calculate MSE
def calculate_mse(model, train_data, true_beta):
    if model == 'GeoERM':
        result = GeoERM(train_data, r=5, link='linear')
        beta_hat = result['step2']
    elif model == 'pERM':
        result = pERM(train_data, link='linear')
        beta_hat = result['step2'] if isinstance(result, dict) else result
    elif model == 'ERM':
        result = ERM(train_data, r=5, link='linear')
    elif model == 'single_task_LR':
        result = single_task_LR(train_data, link='linear')
    elif model == 'pooled_LR':
        result = pooled_LR(train_data, link='linear')
    elif model == 'spectral':
        result = spectral(train_data, r=5, link='linear')
    elif model == 'MoM':
        result = MoM(train_data, r=5)
    elif model == 'AdaptRep':
        result = AdaptRep(train_data, r=5)
    elif model == 'GLasso':
        result = spectral(train_data, r=5, link='linear')
    else:
        raise ValueError("Unknown model")

# Compare the model's beta_hat with the true beta
    beta_hat = result if isinstance(result, np.ndarray) else result['step2']
    mse = max_distance(beta_hat, true_beta)
    return mse

"""n sample

"""

import numpy as np
import json
import os
import sys

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

def main_execution_function(n_tasks=50, n_samples_list=np.arange(60, 201, 20), n_features=50, r=3, epsilon=0.5, h=0.5, n_iterations=1):
    models = ['GeoERM', 'pERM', 'ERM', 'single_task_LR', 'pooled_LR', 'spectral', 'MoM', 'AdaptRep', 'GLasso']

    total_results = {n_samples: {"no_outliers": {model: [] for model in models},
                                 "with_outliers": {model: [] for model in models}}
                     for n_samples in n_samples_list}

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}...")

        for n_samples in n_samples_list:
            print(f"Running for n_samples = {n_samples}...")

            data_no_outliers = generate_data(n=n_samples, p=n_features, r=r, T=n_tasks, epsilon=0, h=h, link='linear')
            data_with_outliers = generate_data(n=n_samples, p=n_features, r=r, T=n_tasks, epsilon=epsilon, h=h, link='linear')

            mse_results_no_outliers = []
            mse_results_with_outliers = []

            for model in models:
                mse_no_outlier = calculate_mse(model, data_no_outliers['data'], data_no_outliers['beta'])
                mse_with_outlier = calculate_mse(model, data_with_outliers['data'], data_with_outliers['beta'])
                mse_results_no_outliers.append(mse_no_outlier)
                mse_results_with_outliers.append(mse_with_outlier)

            for model, mse_no_outlier, mse_with_outlier in zip(models, mse_results_no_outliers, mse_results_with_outliers):
                total_results[n_samples]["no_outliers"][model].append(mse_no_outlier)
                total_results[n_samples]["with_outliers"][model].append(mse_with_outlier)

    averaged_results = {n_samples: {"no_outliers": {model: np.mean(total_results[n_samples]["no_outliers"][model]) for model in models},
                                    "with_outliers": {model: np.mean(total_results[n_samples]["with_outliers"][model]) for model in models}}
                        for n_samples in n_samples_list}

    for n_samples in n_samples_list:
        print(f"Averaged MSE results (no outliers, n_samples={n_samples}):", averaged_results[n_samples]["no_outliers"])
        print(f"Averaged MSE results (with outliers, n_samples={n_samples}):", averaged_results[n_samples]["with_outliers"])

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
if __name__ == "__main__":
    models = ['GeoERM', 'pERM', 'ERM', 'single_task_LR', 'pooled_LR', 'spectral', 'MoM', 'AdaptRep', 'GLasso']
    logging.basicConfig(level=logging.INFO)

    # Call main_execution_function to get both averaged results and standard errors
    averaged_results = main_execution_function(n_iterations=1)
