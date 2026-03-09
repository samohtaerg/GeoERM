# -*- coding: utf-8 -*-
"""
MNIST Pairwise Preprocessing
- 45 pairwise binary classification tasks (C(10,2))
- No corruption types (identity only)
- Balanced sampling: samples_per_class from each digit

Input
-----
MNIST-C dataset (https://github.com/google-research/mnist-c).
Expected directory structure:
    <MNIST_C_PATH>/
        identity/
            train_images.npy
            train_labels.npy

Output
------
A single pickle file: mnist_pairwise_50_45t.pkl
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

# ========== All 45 digit pairs ==========
DIGIT_PAIRS = list(combinations(range(10), 2))  # (0,1), (0,2), ..., (8,9)

# =====================================================

def load_identity_mnist(base_path):
    """Load only the identity (clean) MNIST data."""
    print(f"\nLoading MNIST identity data from {base_path}...")

    img_path = os.path.join(base_path, "identity", "train_images.npy")
    lbl_path = os.path.join(base_path, "identity", "train_labels.npy")

    X = np.load(img_path)
    y = np.load(lbl_path)

    if X.ndim == 4 and X.shape[-1] == 1:
        X = X[..., 0]

    X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0

    print(f"  X shape: {X_flat.shape}, y shape: {y.shape}")
    print(f"  X range: [{X_flat.min():.3f}, {X_flat.max():.3f}]")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X_flat, y


def build_pairwise_data(X, y, digit_pairs, samples_per_class, random_seed):
    """
    Build pairwise classification tasks.
    For each (digit_a, digit_b) pair:
      - Sample `samples_per_class` from each digit
      - Label: digit_a -> 0, digit_b -> 1
    """
    print(f"\nBuilding {len(digit_pairs)} pairwise tasks...")
    print(f"  Samples per class: {samples_per_class} (total per task: {samples_per_class*2})")

    x_list, y_list, task_id_list = [], [], []

    for t, (d_a, d_b) in enumerate(digit_pairs):
        rng = np.random.RandomState(random_seed + t)

        idx_a = np.where(y == d_a)[0]
        idx_b = np.where(y == d_b)[0]

        sampled_a = rng.choice(idx_a, size=samples_per_class, replace=False)
        sampled_b = rng.choice(idx_b, size=samples_per_class, replace=False)

        x_task = np.vstack([X[sampled_a], X[sampled_b]])
        y_task = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])

        shuffle_idx = rng.permutation(len(y_task))
        x_task = x_task[shuffle_idx]
        y_task = y_task[shuffle_idx]

        x_list.append(x_task)
        y_list.append(y_task)
        task_id_list.append(np.full(len(y_task), t, dtype=int))

        if t < 5 or t == len(digit_pairs) - 1:
            print(f"  Task {t:2d} ({d_a} vs {d_b}): {x_task.shape[0]} samples, "
                  f"class 0={int((y_task==0).sum())}, class 1={int((y_task==1).sum())}")

    x_all = np.vstack(x_list)
    y_all = np.concatenate(y_list)
    task_id = np.concatenate(task_id_list)

    print(f"\n  Total: x={x_all.shape}, y={y_all.shape}, tasks={len(digit_pairs)}")

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
    print("MNIST Pairwise Preprocessing (45 tasks)")
    print(f"  {len(DIGIT_PAIRS)} tasks, {SAMPLES_PER_CLASS*2} samples/task")
    print("=" * 70)

    X, y = load_identity_mnist(MNIST_C_PATH)

    x_all, y_all, task_id = build_pairwise_data(
        X, y, DIGIT_PAIRS, SAMPLES_PER_CLASS, RANDOM_SEED
    )

    save_data(x_all, y_all, task_id, OUTPUT_DIR, SAMPLES_PER_CLASS, len(DIGIT_PAIRS))

    print("\n" + "=" * 70)
    print("Preprocessing completed!")
    print(f"  Tasks: {len(DIGIT_PAIRS)}")
    print(f"  Samples per task: {SAMPLES_PER_CLASS*2}")
    print(f"  Total samples: {len(y_all)}")
    print(f"  Feature dim: {x_all.shape[1]} (28x28 = 784)")
    print("=" * 70)
