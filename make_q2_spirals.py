# q2_dataset.py  — long-tail crescents (nonlinear, interleaving)
from dataclasses import dataclass
import numpy as np, torch

@dataclass
class Q2Data:
    X_train: torch.Tensor; y_train: torch.Tensor
    X_val: torch.Tensor;   y_val: torch.Tensor
    X_test: torch.Tensor;  y_test: torch.Tensor

def _rot2d(theta_deg: float):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s],[s,c]], dtype=np.float32)

def make_q2_spirals(
    n_total=2000,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    student_run_seed=42,
):
    gen = np.random.default_rng(123456)  # fixed internal seed

    # ---- Generate two interleaved spirals (non-linear) ----
    n_per_class = n_total // 2
    theta = np.sqrt(gen.random(n_per_class)) * 4 * np.pi  # spiral angle range
    r = theta
    # Spiral 1
    x1 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    # Spiral 2 (shifted by π)
    x2 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)
    X = np.concatenate([x1, x2]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class, dtype=np.int64),
                        np.ones(n_per_class, dtype=np.int64)])

    # Add Gaussian noise
    X += gen.normal(0, 0.15, X.shape).astype(np.float32)

    # ---- Shuffle and split ----
    rng = np.random.default_rng(student_run_seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_train = int(train_ratio * len(X))
    n_val = int(val_ratio * len(X))
    n_test = len(X) - n_train - n_val

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]

    return Q2Data(
        X_train=torch.from_numpy(X_train),
        y_train=torch.from_numpy(y_train),
        X_val=torch.from_numpy(X_val),
        y_val=torch.from_numpy(y_val),
        X_test=torch.from_numpy(X_test),
        y_test=torch.from_numpy(y_test),
    )

import matplotlib.pyplot as plt, torch
# from q2_dataset import make_q2_data
data = make_q2_spirals(student_run_seed=7)
X = torch.cat([data.X_train,data.X_val,data.X_test])
y = torch.cat([data.y_train,data.y_val,data.y_test])
plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", s=20, edgecolor="k", linewidth=0.3)
plt.title("HW1-Q2 Non-linear Dataset"); plt.xlabel("x₁"); plt.ylabel("x₂"); plt.show()
