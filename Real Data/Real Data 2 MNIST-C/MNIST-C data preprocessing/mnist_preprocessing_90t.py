# -*- coding: utf-8 -*-
"""
MNIST Pairwise + Corruption Preprocessing (90 tasks)
- 45 pairwise binary classification tasks (C(10,2)) × 2 corruptions
- Corruptions: identity, brightness
- Total tasks: 90
- Balanced sampling: samples_per_class from each digit

Input
-----
MNIST-C dataset (https://github.com/google-research/mnist-c).
Expected directory structure:
    <MNIST_C_PATH>/
        identity/
            train_images.npy
            train_labels.npy
        brightness/
            train_images.npy
            train_labels.npy

Output
------
A single pickle file: mnist_pairwise_50_90t.pkl
Stored as a tuple (x_all, y_all, task_id_array).
"""

import numpy as np
import pickle
import os
from itertools import combinations

# ============================================================
# USER CONFIGURATION — edit these paths before running
# ============================================================
MNIST_C_PATH = "/path/to/mnist_c"          # root directory of the MNIST-C dataset
OUTPUT_DIR   = "/path/to/output/directory" # where the pkl file will be saved
# ============================================================

# ========== Global Config ==========
SAMPLES_PER_CLASS = 25   # per digit per task → 50 total per task
RANDOM_SEED = 11

CORRUPTION_TYPES = [
    'identity',    # Tasks 0~44
    'brightness',  # Tasks 45~89
]

N_CORRUPTIONS = len(CORRUPTION_TYPES)
N_PAIRS = 45
N_TASKS = N_PAIRS * N_CORRUPTIONS  # 90

# ========== All 45 digit pairs ==========
DIGIT_PAIRS = list(combinations(range(10), 2))


def load_mnist_corruptions(base_path, corruption_types):
    """Load multiple corruptions, return dict of {corr_name: (X_flat, y)}."""
    print(f"\nLoading {len(corruption_types)} corruption(s) from {base_path}...")

    data = {}
    for corr in corruption_types:
        img_path = os.path.join(base_path, corr, "train_images.npy")
        lbl_path = os.path.join(base_path, corr, "train_labels.npy")

        X = np.load(img_path)
        y = np.load(lbl_path)

        if X.ndim == 4 and X.shape[-1] == 1:
            X = X[..., 0]

        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        data[corr] = (X_flat, y)

        print(f"  {corr}: X={X_flat.shape}, y={y.shape}, "
              f"range=[{X_flat.min():.3f}, {X_flat.max():.3f}]")

    return data


def build_pairwise_data(data, digit_pairs, corruption_types, samples_per_class, random_seed):
    """
    Build pairwise × corruption tasks.
    task_id = corr_idx * 45 + pair_idx
    """
    n_pairs = len(digit_pairs)
    n_corruptions = len(corruption_types)
    total_tasks = n_pairs * n_corruptions

    print(f"\nBuilding {n_corruptions} corruptions × {n_pairs} pairs = {total_tasks} tasks...")
    print(f"  Samples per class: {samples_per_class} (total per task: {samples_per_class*2})")

    x_list, y_list, task_id_list = [], [], []

    for c_idx, corr in enumerate(corruption_types):
        X_corr, y_corr = data[corr]

        for p_idx, (d_a, d_b) in enumerate(digit_pairs):
            task_id = c_idx * n_pairs + p_idx
            rng = np.random.RandomState(random_seed + task_id)

            idx_a = np.where(y_corr == d_a)[0]
            idx_b = np.where(y_corr == d_b)[0]

            sampled_a = rng.choice(idx_a, size=samples_per_class, replace=False)
            sampled_b = rng.choice(idx_b, size=samples_per_class, replace=False)

            x_task = np.vstack([X_corr[sampled_a], X_corr[sampled_b]])
            y_task = np.concatenate([np.zeros(samples_per_class),
                                     np.ones(samples_per_class)])

            shuffle_idx = rng.permutation(len(y_task))
            x_task = x_task[shuffle_idx]
            y_task = y_task[shuffle_idx]

            x_list.append(x_task)
            y_list.append(y_task)
            task_id_list.append(np.full(len(y_task), task_id, dtype=int))

            if p_idx < 3 or p_idx == n_pairs - 1:
                print(f"  Task {task_id:3d} [{corr}] ({d_a} vs {d_b}): "
                      f"class 0={int((y_task==0).sum())}, class 1={int((y_task==1).sum())}")

        print(f"  ... [{c_idx+1}/{n_corruptions}] {corr} done "
              f"(tasks {c_idx*n_pairs}~{(c_idx+1)*n_pairs-1})\n")

    x_all = np.vstack(x_list)
    y_all = np.concatenate(y_list)
    task_id = np.concatenate(task_id_list)

    print(f"  Total: x={x_all.shape}, y={y_all.shape}, unique tasks={np.unique(task_id).shape[0]}")

    return x_all, y_all, task_id


def save_data(x, y, task_id, output_dir, samples_per_class, n_tasks):
    """Save processed data as a single pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"mnist_pairwise_{samples_per_class*2}_{n_tasks}t.pkl"
    save_path = os.path.join(output_dir, filename)

    with open(save_path, 'wb') as f:
        pickle.dump((x, y, task_id), f)

    print(f"\n  Saved to {save_path}")
    print(f"  File size: {os.path.getsize(save_path)/1024/1024:.1f} MB")

    return save_path


if __name__ == '__main__':

    print("=" * 70)
    print("MNIST Pairwise + Corruption Preprocessing (90 tasks)")
    print(f"  {len(DIGIT_PAIRS)} pairs × {N_CORRUPTIONS} corruptions = {N_TASKS} tasks")
    print(f"  Corruptions: {CORRUPTION_TYPES}")
    print(f"  Samples per task: {SAMPLES_PER_CLASS*2}")
    print("=" * 70)

    data = load_mnist_corruptions(MNIST_C_PATH, CORRUPTION_TYPES)

    x_all, y_all, task_id = build_pairwise_data(
        data, DIGIT_PAIRS, CORRUPTION_TYPES, SAMPLES_PER_CLASS, RANDOM_SEED
    )

    save_data(x_all, y_all, task_id, OUTPUT_DIR, SAMPLES_PER_CLASS, N_TASKS)

    print("\n" + "=" * 70)
    print("Preprocessing completed!")
    print(f"  Tasks: {N_TASKS} ({N_PAIRS} pairs × {N_CORRUPTIONS} corruptions)")
    print(f"  Samples per task: {SAMPLES_PER_CLASS*2}")
    print(f"  Total samples: {len(y_all)}")
    print(f"  Feature dim: {x_all.shape[1]} (28x28 = 784)")
    print("=" * 70)
