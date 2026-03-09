# -*- coding: utf-8 -*-
"""
MNIST Pairwise + Corruption + Adversarial Preprocessing (92 tasks)
- 45 pairwise binary classification tasks × 2 corruptions = 90 tasks
- Plus 2 adversarial tasks (randomly shuffled labels)
- Total tasks: 92

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
A single pickle file: mnist_pairwise_50_92t_adv.pkl
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
N_ADVERSARIAL = 2        # number of adversarial tasks to append

CORRUPTION_TYPES = [
    'identity',    # Tasks 0~44
    'brightness',  # Tasks 45~89
]

N_CORRUPTIONS = len(CORRUPTION_TYPES)
N_PAIRS = 45
ORIGINAL_TASKS = N_PAIRS * N_CORRUPTIONS   # 90
TOTAL_TASKS = ORIGINAL_TASKS + N_ADVERSARIAL  # 92

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
        print(f"  {corr}: X={X_flat.shape}, y={y.shape}")
    return data


def build_pairwise_data(data, digit_pairs, corruption_types, samples_per_class, random_seed):
    """Build original pairwise × corruption tasks."""
    n_pairs = len(digit_pairs)
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

    x_all = np.vstack(x_list)
    y_all = np.concatenate(y_list)
    task_id = np.concatenate(task_id_list)
    return x_all, y_all, task_id


def add_adversarial_tasks(x_all, y_all, task_id, n_adversarial, random_seed):
    """
    Append adversarial tasks by copying existing tasks and randomly
    shuffling their labels, making them pure noise.
    """
    print(f"\nAdding {n_adversarial} adversarial tasks (labels shuffled)...")
    rng_adv = np.random.RandomState(random_seed + 999)
    existing_tids = np.unique(task_id)
    adv_source_tasks = rng_adv.choice(existing_tids, size=n_adversarial, replace=False)
    new_tid_start = existing_tids.max() + 1

    x_adv_list, y_adv_list, tid_adv_list = [], [], []

    for i, src_tid in enumerate(adv_source_tasks):
        idx = np.where(task_id == src_tid)[0]
        x_adv = x_all[idx].copy()
        y_adv = y_all[idx].copy()
        rng_adv.shuffle(y_adv)   # fully randomise labels

        new_tid = new_tid_start + i
        x_adv_list.append(x_adv)
        y_adv_list.append(y_adv)
        tid_adv_list.append(np.full(len(y_adv), new_tid, dtype=int))
        print(f"  Adversarial task {new_tid}: copied from source task {src_tid}, labels shuffled")

    x_final = np.vstack([x_all] + x_adv_list)
    y_final = np.concatenate([y_all] + y_adv_list)
    tid_final = np.concatenate([task_id] + tid_adv_list)

    return x_final, y_final, tid_final


def save_data(x, y, task_id, output_dir, samples_per_class, total_tasks):
    """Save processed data as a single pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"mnist_pairwise_{samples_per_class*2}_{total_tasks}t_adv.pkl"
    save_path = os.path.join(output_dir, filename)
    with open(save_path, 'wb') as f:
        pickle.dump((x, y, task_id), f)
    print(f"\n  Saved to {save_path}")
    print(f"  File size: {os.path.getsize(save_path)/1024/1024:.1f} MB")
    return save_path


if __name__ == '__main__':
    print("=" * 70)
    print("MNIST Pairwise + Corruption + Adversarial Preprocessing (92 tasks)")
    print(f"  {ORIGINAL_TASKS} clean tasks + {N_ADVERSARIAL} adversarial tasks = {TOTAL_TASKS} total")
    print(f"  Corruptions: {CORRUPTION_TYPES}")
    print("=" * 70)

    data = load_mnist_corruptions(MNIST_C_PATH, CORRUPTION_TYPES)

    x_all, y_all, task_id = build_pairwise_data(
        data, DIGIT_PAIRS, CORRUPTION_TYPES, SAMPLES_PER_CLASS, RANDOM_SEED
    )

    x_final, y_final, tid_final = add_adversarial_tasks(
        x_all, y_all, task_id, N_ADVERSARIAL, RANDOM_SEED
    )

    save_data(x_final, y_final, tid_final, OUTPUT_DIR, SAMPLES_PER_CLASS, TOTAL_TASKS)

    print("\n" + "=" * 70)
    print("Preprocessing completed!")
    print(f"  Final tasks: {len(np.unique(tid_final))}")
    print(f"  Total samples: {len(y_final)}")
    print(f"  Feature dim: {x_final.shape[1]} (28x28 = 784)")
    print("=" * 70)
