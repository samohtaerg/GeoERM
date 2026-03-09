# -*- coding: utf-8 -*-
"""GeoERM MNIST Pairwise 45t - LibMTL HPS + EW / PCGrad

Run on Google Colab with a T4 GPU.
Update DATA_PATH and OUTPUT_DIR in the __main__ block before running.
"""

# Setup and patch (run once at top of Colab)
import subprocess
subprocess.run(['pip', 'install', 'LibMTL', '--quiet'], check=True)

import sys, os, re
for path in sys.path:
    p1 = os.path.join(path, 'LibMTL', 'model', 'resnet.py')
    if os.path.exists(p1):
        txt = open(p1).read().replace(
            'from torchvision.models.utils import load_state_dict_from_url',
            'from torch.hub import load_state_dict_from_url'
        )
        open(p1, 'w').write(txt)
        print("Patched: resnet.py")
    p2 = os.path.join(path, 'LibMTL', 'trainer.py')
    if os.path.exists(p2):
        txt = open(p2).read().replace('loader[1].next()', 'next(loader[1])')
        open(p2, 'w').write(txt)
        print("Patched: trainer.py")
    if os.path.exists(p1):
        break

p3 = os.path.join(path, 'LibMTL', 'weighting', 'abstract_weighting.py')
if os.path.exists(p3):
    txt = open(p3).read()
    old_func = re.search(r'    def _reset_grad\(self.*?(?=\n    def |\nclass |\Z)', txt, re.DOTALL)
    if old_func:
        new_func = '''    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is None:
                count += 1
                continue
            beg = 0 if count == 0 else sum(self.grad_index[:count])
            end = sum(self.grad_index[:(count+1)])
            param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1'''
        txt = txt[:old_func.start()] + new_func + txt[old_func.end():]
        open(p3, 'w').write(txt)
        print("Patched: abstract_weighting.py")

import csv, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from LibMTL import Trainer
from LibMTL.architecture import HPS
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric
from LibMTL.weighting import EW, PCGrad
from LibMTL.utils import set_random_seed


class MNISTEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            d = h
        self.encoder = nn.Sequential(*layers)
        self.rep_dim = hidden_dims[-1]

    def forward(self, x):
        return self.encoder(x)


class MNISTDecoder(nn.Module):
    def __init__(self, rep_dim=64, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(rep_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


class MNISTTaskDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def run(seed=0, epochs=100, batch_size=16, lr=1e-3, weighting='EW', data_path=None):
    set_random_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Seed {seed} | Device: {device} | Weighting: {weighting}")

    with open(data_path, 'rb') as f:
        x, y, task_id_array = pickle.load(f)

    train_sets, test_sets = [], []
    for t in np.unique(task_id_array):
        idx = np.where(task_id_array == t)[0]
        np.random.shuffle(idx)
        m = len(idx) // 2
        train_sets.append((x[idx[:m]], y[idx[:m]]))
        test_sets.append((x[idx[m:]], y[idx[m:]]))

    T = len(train_sets)
    input_dim = x.shape[1]
    task_names = [f"task_{i}" for i in range(T)]
    print(f"Tasks: {T} | Input dim: {input_dim}")

    task_dict = {
        t: {
            'metrics':    ['ACC'],
            'metrics_fn': AccMetric(),
            'loss_fn':    CELoss(),
            'weight':     [1]
        } for t in task_names
    }

    train_loaders = {
        task_names[i]: DataLoader(
            MNISTTaskDataset(*train_sets[i]),
            batch_size=batch_size, shuffle=True, drop_last=True
        ) for i in range(T)
    }
    test_loaders = {
        task_names[i]: DataLoader(
            MNISTTaskDataset(*test_sets[i]),
            batch_size=batch_size, shuffle=False
        ) for i in range(T)
    }

    def encoder_class():
        return MNISTEncoder(input_dim=input_dim)

    decoders = nn.ModuleDict({
        t: MNISTDecoder(rep_dim=64, num_classes=2)
        for t in task_names
    })

    weighting_cls = {'EW': EW, 'PCGrad': PCGrad}[weighting]

    trainer = Trainer(
        task_dict       = task_dict,
        weighting       = weighting_cls,
        architecture    = HPS,
        encoder_class   = encoder_class,
        decoders        = decoders,
        rep_grad        = False,
        multi_input     = True,
        optim_param     = {'optim': 'adam', 'lr': lr, 'weight_decay': 1e-4},
        scheduler_param = {'scheduler': 'step', 'step_size': 30, 'gamma': 0.5},
        arch_args       = {},
        weight_args     = {},
        device          = device
    )

    trainer.train(train_loaders, test_loaders, epochs)

    trainer.model.eval()
    errors = []
    with torch.no_grad():
        for i, t in enumerate(task_names):
            xte, yte = test_sets[i]
            xte = torch.FloatTensor(xte).to(device)
            logits = trainer.model(xte)[t]
            pred = logits.argmax(1).cpu().numpy()
            errors.append(np.mean(pred != yte))

    errors = np.array(errors)
    print(f"Mean error: {errors.mean():.4f}")
    return errors


if __name__ == '__main__':
    from google.colab import drive
    drive.mount('/content/drive')

    # ── USER CONFIGURATION ──────────────────────────────────────────────────
    DATA_PATH  = '/content/drive/MyDrive/MNIST Data 3 Setups/mnist_pairwise_50_45t.pkl'
    OUTPUT_DIR = '/content/drive/MyDrive/deepMTLoutput/mnist_45t_libmtl_hps'
    # ────────────────────────────────────────────────────────────────────────

    NUM_SEEDS = 100
    EPOCHS    = 100
    R         = 10
    METHODS   = ['EW', 'PCGrad']

    for METHOD in METHODS:
        method_dir = os.path.join(OUTPUT_DIR, METHOD)
        os.makedirs(method_dir, exist_ok=True)

        for seed in range(NUM_SEEDS):
            print(f"\n{'='*50}")
            print(f"Method: {METHOD} | Replication {seed+1}/{NUM_SEEDS}")
            print(f"{'='*50}")

            t_start = time.time()
            errors = run(seed=seed, epochs=EPOCHS, weighting=METHOD, data_path=DATA_PATH)
            runtime = time.time() - t_start

            out_file = os.path.join(method_dir, f'libmtl_{METHOD}_HPS_{R}_{seed}.csv')
            with open(out_file, 'w', newline='') as f:
                csv.writer(f).writerow(list(errors) + [runtime])

            print(f"Runtime: {runtime:.1f}s | Saved to {out_file}")

    print("\nDone.")
