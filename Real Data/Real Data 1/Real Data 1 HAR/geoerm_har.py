#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoERM HAR Dataset MTL Experiment
===================================
Unified script for all rank values used in the JCGS paper.
30 subjects treated as tasks; binary activity classification.

Usage
-----
    python geoerm_har.py --r 5
    python geoerm_har.py --r 10
    python geoerm_har.py --r 15

The seed is read from $SLURM_ARRAY_TASK_ID (defaults to 0 for local runs).

Methods compared (rows of output CSV)
--------------------------------------
  0 : Single-task
  1 : Pooled
  2 : ERM
  3 : ARMUL
  4 : pERM
  5 : GeoERM  (optimized v2, orthogonal-rejection penalty)
"""

import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import LinearRegression, LogisticRegression

# ============================================================
# USER CONFIGURATION — edit these paths before running
# ============================================================
ROOT_PATH     = "/path/to/RL_TL_MTL/"        # root of the RL_TL_MTL repository
HAR_DATA_PATH = "/path/to/har_standardized.pkl"  # preprocessed HAR data (see UCI HAR Dataset/har_preprocessing.py)
OUTPUT_ROOT   = "/path/to/output/directory"   # directory where result CSVs will be saved
# ============================================================

sys.path.append(ROOT_PATH)
for _sub in ["benchmarks/ARMUL", "benchmarks/AdaptRep", "benchmarks/GLasso"]:
    sys.path.append(os.path.join(ROOT_PATH, _sub))
sys.path.append(os.path.join(ROOT_PATH, "benchmarks/ARMUL/ARMUL"))

from mtl_func_torch import pERM, ERM, single_task_LR, pooled_LR   # noqa: E402
from funcs import prediction, all_classification_error              # noqa: E402
from ARMUL.ARMUL import ARMUL_blackbox                              # noqa: E402

# ---------------------------------------------------------------------------
# GeoERM (optimized v2) – orthogonal-rejection Stiefel penalty
# ---------------------------------------------------------------------------

def polar_retraction(X):
    U, _, Vt = torch.linalg.svd(X, full_matrices=False)
    return U @ Vt


def projection(X, U):
    XtU = torch.matmul(X.transpose(-2, -1), U)
    symXtU = (XtU + XtU.transpose(-2, -1)) / 2
    return U - torch.matmul(X, symXtU)


def column_norm(A):
    return np.array([np.linalg.norm(A[:, j]) for j in range(A.shape[1])])


def initialize_stiefel_matrix(p, r):
    Q, _ = np.linalg.qr(np.random.randn(p, r))
    return Q[:, :r]


def initialize_stiefel_tensor(T, p, r):
    A = np.zeros((T, p, r))
    for t in range(T):
        Q, _ = np.linalg.qr(np.random.randn(p, r))
        A[t] = Q[:, :r]
    return A


def stiefel_penalty(A_t, A_bar):
    """
    ||A_t A_t^T - A_bar A_bar^T||_2  via orthogonal rejection.

    Q = A_t - A_bar (A_bar^T A_t) is the component of A_t orthogonal to
    the column space of A_bar.  Its largest singular value equals
    sin(theta_max) = ||P_{A_t} - P_{A_bar}||_2.
    Complexity O(pr^2); no sqrt / clamp; stable gradients.
    """
    M = torch.matmul(A_bar.T, A_t)
    Q = A_t - torch.matmul(A_bar, M)
    return torch.linalg.svdvals(Q)[0]


def GeoERM(data, r=3, T1=1, T2=1, R=None, r_bar=None,
           lr=0.01, max_iter=20, C1=1, C2=1,
           delta=0.05, adaptive=False, info=False, tol=1e-6,
           link="linear"):
    """Two-step penalized GeoERM on the Stiefel manifold (optimized v2)."""
    if info:
        print("GeoERM starts running...", flush=True)

    T = len(data)
    n = np.array([x.shape[0] for (x, _) in data])
    p = data[0][0].shape[1]
    n_total = int(n.sum())

    x_np = np.zeros((n_total, p))
    y_np = np.zeros(n_total)
    task_range = []
    idx = 0
    for t in range(T):
        task_range.append(range(idx, idx + n[t]))
        x_np[task_range[t]] = data[t][0]
        y_np[task_range[t]] = data[t][1]
        idx += n[t]

    if adaptive:
        beta_st = np.zeros((p, T))
        for t in range(T):
            if link == "linear":
                beta_st[:, t] = LinearRegression(fit_intercept=False).fit(
                    x_np[task_range[t]], y_np[task_range[t]]).coef_
            else:
                beta_st[:, t] = LogisticRegression(fit_intercept=False).fit(
                    x_np[task_range[t]], y_np[task_range[t]]).coef_
        norms = column_norm(beta_st)
        if R is None:
            R = np.median(norms) * 2
        for t in range(T):
            if norms[t] > R:
                beta_st[:, t] *= R / norms[t]
        if r_bar is None:
            r_bar = p
        threshold = (T1 * np.sqrt((p + np.log(T)) / n.max())
                     + T2 * R * r_bar ** (-3/4))
        r = int(np.where(np.linalg.svd(beta_st / np.sqrt(T))[1] > threshold)[0].max()) + 1
        if info:
            print(f"Selected r = {r}")

    y = torch.tensor(y_np, requires_grad=False)
    x = torch.tensor(x_np, requires_grad=False)

    A_hat = initialize_stiefel_tensor(T, p, r)
    A_bar = initialize_stiefel_matrix(p, r)
    A_bar[:r, :r] = np.eye(r)
    for t in range(T):
        A_hat[t, :r, :r] = np.eye(r)
    theta_hat = np.zeros((r, T))

    A_bar     = torch.tensor(A_bar,     requires_grad=True,  dtype=torch.float64)
    A_hat     = torch.tensor(A_hat,     requires_grad=True,  dtype=torch.float64)
    theta_hat = torch.tensor(theta_hat, requires_grad=True,  dtype=torch.float64)

    lam = np.sqrt(r * (p + np.log(T))) * C1

    if link == "linear":
        def ftotal(A, theta, Ab):
            s = 0
            for t in range(T):
                res = y[task_range[t]] - x[task_range[t]] @ A[t] @ theta[:, t]
                s += (1 / (2 * n_total)) * torch.dot(res, res) \
                   + lam * np.sqrt(n[t]) / n_total * stiefel_penalty(A[t], Ab)
            return s
    else:
        def ftotal(A, theta, Ab):
            s = 0
            for t in range(T):
                logits = x[task_range[t]] @ A[t] @ theta[:, t]
                s += (1 / n_total) * torch.dot(1 - y[task_range[t]], logits) \
                   + (1 / n_total) * torch.sum(torch.log(1 + torch.exp(-logits))) \
                   + lam * np.sqrt(n[t]) / n_total * stiefel_penalty(A[t], Ab)
            return s

    opt1 = optim.Adam([A_hat, theta_hat, A_bar], lr=lr)
    loss_last = 1e8
    for i in range(max_iter):
        opt1.zero_grad()
        loss = ftotal(A_hat, theta_hat, A_bar)
        loss.backward()
        with torch.no_grad():
            A_hat.grad.copy_(projection(A_hat, A_hat.grad))
            A_bar.grad.copy_(projection(A_bar, A_bar.grad))
        opt1.step()
        with torch.no_grad():
            A_hat.copy_(polar_retraction(A_hat))
            A_bar.copy_(polar_retraction(A_bar))
        if info and (i + 1) % 100 == 0:
            print(f"  [Step 1] iter {i+1}/{max_iter}, loss={loss.item():.6f}", flush=True)
        if abs(loss_last - loss.item()) / loss.item() <= tol:
            if info:
                print("  Converged early.", flush=True)
            break
        loss_last = loss.item()

    beta_step1 = torch.zeros(p, T, dtype=torch.float64)
    for t in range(T):
        beta_step1[:, t] = A_hat[t] @ theta_hat[:, t]
    beta_step1 = beta_step1.detach()
    if info:
        print("Step 1 complete.\n", flush=True)

    gamma = np.sqrt(p + np.log(T)) * C2
    beta = torch.zeros(p, T, requires_grad=True, dtype=torch.float64)

    if link == "linear":
        def ftotal2(b):
            s = 0
            for t in range(T):
                res = y[task_range[t]] - x[task_range[t]] @ b[:, t]
                s += (1 / (2 * n[t])) * torch.dot(res, res) \
                   + gamma / np.sqrt(n[t]) * torch.norm(b[:, t] - beta_step1[:, t])
            return s
    else:
        def ftotal2(b):
            s = 0
            for t in range(T):
                logits = x[task_range[t]] @ b[:, t]
                s += (1 / n[t]) * torch.dot(1 - y[task_range[t]], logits) \
                   + (1 / n[t]) * torch.sum(torch.log(1 + torch.exp(-logits))) \
                   + gamma / np.sqrt(n[t]) * torch.norm(b[:, t] - beta_step1[:, t])
            return s

    opt2 = optim.Adam([beta], lr=lr)
    loss_last = 1e8
    for i in range(max_iter):
        opt2.zero_grad()
        loss2 = ftotal2(beta)
        loss2.backward()
        opt2.step()
        if info and (i + 1) % 100 == 0:
            print(f"  [Step 2] iter {i+1}/{max_iter}, loss={loss2.item():.6f}", flush=True)
        if abs(loss_last - loss2.item()) / loss2.item() <= tol:
            if info:
                print("  Converged early.", flush=True)
            break
        loss_last = loss2.item()

    if info:
        print("Step 2 complete.\nGeoERM done.", flush=True)

    return {
        "step1": beta_step1.numpy(),
        "step2": beta.detach().numpy(),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_sim(train_data, test_data, r, seed):
    n_tasks = len(train_data)
    est_error = np.zeros((6, n_tasks))
    runtime   = np.full(6, np.nan)

    def _run(name, idx, fn):
        print(f"Running {name}...", flush=True)
        t0 = time.time()
        result = fn()
        runtime[idx] = time.time() - t0
        print(f"  {name} done in {runtime[idx]:.2f}s.", flush=True)
        return result

    # 0 – Single-task
    bhat = _run("Single-task", 0, lambda: single_task_LR(train_data, link="logistic"))
    est_error[0] = all_classification_error(prediction(bhat, test_data), test_data)

    # 1 – Pooled
    bhat = _run("Pooled", 1, lambda: pooled_LR(train_data, link="logistic"))
    est_error[1] = all_classification_error(prediction(bhat, test_data), test_data)

    # 2 – ERM
    bhat = _run("ERM", 2, lambda: ERM(data=train_data, r=r, info=False, link="logistic"))
    est_error[2] = all_classification_error(prediction(bhat, test_data), test_data)

    # 3 – ARMUL (allowed to fail)
    try:
        bhat = _run("ARMUL", 3, lambda: ARMUL_blackbox(
            train_data, r, eta=0.1, L=10, n_fold=5, seed=seed,
            link="logistic", c_max=5))
        est_error[3] = all_classification_error(prediction(bhat, test_data), test_data)
    except Exception as e:
        print(f"  ARMUL failed: {e}", flush=True)

    # 4 – pERM
    bhat = _run("pERM", 4, lambda: pERM(
        data=train_data, r=r, C1=1, C2=0.5,
        adaptive=False, link="logistic", max_iter=1000, info=False))
    est_error[4] = all_classification_error(prediction(bhat["step2"], test_data), test_data)

    # 5 – GeoERM (optimized v2)
    bhat = _run("GeoERM (v2)", 5, lambda: GeoERM(
        data=train_data, r=r, C1=1, C2=0.5,
        adaptive=False, link="logistic", max_iter=2000, info=False))
    est_error[5] = all_classification_error(prediction(bhat["step2"], test_data), test_data)

    return est_error, runtime


def main():
    parser = argparse.ArgumentParser(
        description="GeoERM HAR MTL experiment")
    parser.add_argument("--r", type=int, required=True,
                        help="Rank of the shared subspace (e.g. 5, 10, 15)")
    parser.add_argument("--test_ratio", type=float, default=0.5,
                        help="Fraction of each subject's data held out for testing")
    parser.add_argument("--data_path", type=str, default=HAR_DATA_PATH,
                        help="Path to har_standardized.pkl")
    parser.add_argument("--out_root", type=str, default=OUTPUT_ROOT,
                        help="Root directory for output CSVs")
    args = parser.parse_args()

    seed = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Config : r={args.r}, seed={seed}", flush=True)
    print(f"Loading: {args.data_path}", flush=True)

    with open(args.data_path, "rb") as f:
        (x_pca, y, subject) = pickle.load(f)
    print(f"Loaded : x={x_pca.shape}, y={y.shape}", flush=True)

    train_data, test_data = [], []
    for t in np.unique(subject):
        idx = np.where(subject == t)[0]
        n_test = int(np.floor(idx.size * args.test_ratio))
        test_idx  = np.random.choice(idx, size=n_test, replace=False)
        train_idx = np.setdiff1d(idx, test_idx)
        train_data.append((x_pca[train_idx], y[train_idx]))
        test_data.append( (x_pca[test_idx],  y[test_idx]))

    output_dir = os.path.join(args.out_root, f"har_r{args.r}_geo2_complete")
    os.makedirs(output_dir, exist_ok=True)
    file_path    = os.path.join(output_dir, f"realdata_{args.r}_{seed}.csv")
    runtime_path = os.path.join(output_dir, f"realdata_{args.r}_{seed}_runtime.csv")

    if os.path.exists(file_path):
        print("Output already exists. Exiting.", flush=True)
        sys.exit(0)

    print("Starting experiment...", flush=True)
    est_error, runtime = run_sim(train_data, test_data, args.r, seed)

    with open(file_path, "w", newline="") as f:
        csv.writer(f).writerows(est_error)
    with open(runtime_path, "w", newline="") as f:
        csv.writer(f).writerow(runtime.tolist())

    print(f"Saved : {file_path}", flush=True)
    print(f"Saved : {runtime_path}", flush=True)


if __name__ == "__main__":
    main()
