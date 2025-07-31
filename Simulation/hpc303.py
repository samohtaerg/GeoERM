# -*- coding: utf-8 -*-
"""# 9 Models Linear Setup

Updated Setup with iteration
"""

from joblib import Parallel, delayed
import numpy as np


# Helper function to calculate MSE
def calculate_mse(model, train_data, true_beta):
    if model == 'GeoERM':
        result = GeoERM(train_data, r=3, link='linear')
        beta_hat = result['step2']
    elif model == 'pERM':
        result = pERM(train_data, r=3, link='linear')
        beta_hat = result['step2'] if isinstance(result, dict) else result
    elif model == 'ERM':
        result = ERM(train_data, r=3, link='linear')
    elif model == 'single_task_LR':
        result = single_task_LR(train_data, link='linear')
    elif model == 'pooled_LR':
        result = pooled_LR(train_data, link='linear')
    elif model == 'spectral':
        result = spectral(train_data, r=3, link='linear')
    elif model == 'MoM':
        result = MoM(train_data, r=3)
    elif model == 'AdaptRep':
        result = AdaptRep(train_data, r=3)
    elif model == 'GLasso':
        result = spectral(train_data, r=3, link='linear')
    else:
        raise ValueError("Unknown model")

    # Compare the model's beta_hat with the true beta
    # beta_hat = result if isinstance(result, np.ndarray) else result['step2']
    # mse = np.mean((beta_hat - true_beta)**2)
    # Compare the model's beta_hat with the true beta
    beta_hat = result if isinstance(result, np.ndarray) else result['step2']
    mse = max_distance(beta_hat, true_beta)
    return mse

def main_execution_function(n_tasks=50, n_samples=100, n_features=30, r=3, epsilon=0.1, h_list=np.arange(0.1, 0.9, 0.1), n_iterations=1):
    # Initialize total results to accumulate over iterations
    total_results = {h: {"no_outliers": {model: [] for model in ['GeoERM', 'pERM', 'ERM', 'single_task_LR', 'pooled_LR', 'spectral', 'MoM', 'AdaptRep', 'GLasso']},
                         "with_outliers": {model: [] for model in ['GeoERM', 'pERM', 'ERM', 'single_task_LR', 'pooled_LR', 'spectral', 'MoM', 'AdaptRep', 'GLasso']}}
                     for h in h_list}

    # Loop over each iteration
    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}...")

        # Loop over each h
        for h in h_list:
            print(f"Running for h = {h}...")

            # Generate datasets with and without outliers, for each h
            data_no_outliers = generate_data(n=n_samples, p=n_features, r=r, T=n_tasks, epsilon=0, h=h, link='linear')
            data_with_outliers = generate_data(n=n_samples, p=n_features, r=r, T=n_tasks, epsilon=epsilon, h=h, link='linear')

            # Models
            models = ['GeoERM', 'pERM', 'ERM', 'single_task_LR', 'pooled_LR', 'spectral', 'MoM', 'AdaptRep', 'GLasso']

            # Sequential execution
            mse_results_no_outliers = []
            mse_results_with_outliers = []

            for model in models:
                mse_no_outlier = calculate_mse(model, data_no_outliers['data'], data_no_outliers['beta'])
                mse_with_outlier = calculate_mse(model, data_with_outliers['data'], data_with_outliers['beta'])
                mse_results_no_outliers.append(mse_no_outlier)
                mse_results_with_outliers.append(mse_with_outlier)

            # Accumulate the results
            for model, mse_no_outlier, mse_with_outlier in zip(models, mse_results_no_outliers, mse_results_with_outliers):
                total_results[h]["no_outliers"][model].append(mse_no_outlier)
                total_results[h]["with_outliers"][model].append(mse_with_outlier)

    # After all iterations, compute the average and standard error
    averaged_results = {h: {"no_outliers": {model: np.mean(total_results[h]["no_outliers"][model]) for model in models},
                            "with_outliers": {model: np.mean(total_results[h]["with_outliers"][model]) for model in models}}
                        for h in h_list}

    standard_errors = {h: {"no_outliers": {model: np.std(total_results[h]["no_outliers"][model], ddof=1) / np.sqrt(n_iterations) for model in models},
                           "with_outliers": {model: np.std(total_results[h]["with_outliers"][model], ddof=1) / np.sqrt(n_iterations) for model in models}}
                       for h in h_list}

    # Print averaged results with standard error
    for h in h_list:
        print(f"Averaged MSE results (no outliers, h={h}):", averaged_results[h]["no_outliers"])
        print(f"Standard Errors (no outliers, h={h}):", standard_errors[h]["no_outliers"])
        print(f"Averaged MSE results (with outliers, h={h}):", averaged_results[h]["with_outliers"])
        print(f"Standard Errors (with outliers, h={h}):", standard_errors[h]["with_outliers"])

    import os
    import json
    import sys

    output_dir = ''

    os.makedirs(output_dir, exist_ok=True)

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")  
    averaged_file_path = os.path.join(output_dir, f'averaged_results_{task_id}.json')
    standard_errors_file_path = os.path.join(output_dir, f'standard_errors_{task_id}.json')

    if os.path.exists(averaged_file_path) or os.path.exists(standard_errors_file_path):
        print(f"File {averaged_file_path} or {standard_errors_file_path} exists. Stopping the script.", flush=True)
        sys.exit(0)
    else:
        print(f"Files do not exist. Continuing the script.", flush=True)

    with open(averaged_file_path, 'w') as f:
        json.dump(averaged_results, f, indent=4)
    with open(standard_errors_file_path, 'w') as f:
        json.dump(standard_errors, f, indent=4)

    print(f"Results saved to {averaged_file_path} and {standard_errors_file_path}", flush=True)

    return averaged_results, standard_errors
    
if __name__ == "__main__":
    models = ['GeoERM', 'pERM', 'ERM', 'single_task_LR', 'pooled_LR', 'spectral', 'MoM', 'AdaptRep', 'GLasso']
    logging.basicConfig(level=logging.INFO)

    # Call main_execution_function to get both averaged results and standard errors
    averaged_results, standard_errors = main_execution_function(n_iterations=1)
