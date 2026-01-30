"""
Generalization study in neural networks.

This script looks at how model capacity and training data size
affect generalization. The focus is on patterns, not just accuracy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# experiment settings

EPOCHS = 400
LR = 0.01
RUNS = 5


# data creation

def generate_data(samples, noise=20.0):
    X, y = make_regression(
        n_samples=samples,
        n_features=10,
        noise=noise,
        random_state=42
    )

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, y


# neural network

class SimpleNN(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):
        return self.net(x)


# training function

def train_model(X_train, y_train, X_test, y_test, hidden_units, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleNN(hidden_units)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    train_mse = mean_squared_error(
        y_train.numpy(),
        model(X_train).detach().numpy()
    )

    test_mse = mean_squared_error(
        y_test.numpy(),
        model(X_test).detach().numpy()
    )

    return train_mse, test_mse


# Experiment 1: model capacity

hidden_sizes = [2, 5, 10, 50, 100]

X, y = generate_data(2000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_mean, train_std = [], []
test_mean, test_std = [], []

for h in hidden_sizes:
    train_errors = []
    test_errors = []

    for seed in range(RUNS):
        tr, te = train_model(
            X_train, y_train, X_test, y_test, h, seed
        )
        train_errors.append(tr)
        test_errors.append(te)

    train_mean.append(np.mean(train_errors))
    train_std.append(np.std(train_errors))
    test_mean.append(np.mean(test_errors))
    test_std.append(np.std(test_errors))


plt.figure()
plt.plot(hidden_sizes, train_mean, marker='o', label='Training error')
plt.fill_between(
    hidden_sizes,
    np.array(train_mean) - np.array(train_std),
    np.array(train_mean) + np.array(train_std),
    alpha=0.3
)

plt.plot(hidden_sizes, test_mean, marker='o', label='Test error')
plt.fill_between(
    hidden_sizes,
    np.array(test_mean) - np.array(test_std),
    np.array(test_mean) + np.array(test_std),
    alpha=0.3
)

plt.xlabel("Hidden units")
plt.ylabel("Mean squared error")
plt.title("Model capacity vs generalization")
plt.legend()
plt.grid(True)
plt.savefig("capacity_vs_generalization.png", dpi=300)
plt.show()


# Experiment 2: training data size

data_sizes = [50, 150, 500, 1000, 1800]

train_curve_mean, train_curve_std = [], []
test_curve_mean, test_curve_std = [], []

for size in data_sizes:
    X, y = generate_data(size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_errors = []
    test_errors = []

    for seed in range(RUNS):
        tr, te = train_model(
            X_train, y_train, X_test, y_test, hidden_units=50, seed=seed
        )
        train_errors.append(tr)
        test_errors.append(te)

    train_curve_mean.append(np.mean(train_errors))
    train_curve_std.append(np.std(train_errors))
    test_curve_mean.append(np.mean(test_errors))
    test_curve_std.append(np.std(test_errors))


plt.figure()
plt.errorbar(
    data_sizes,
    train_curve_mean,
    yerr=train_curve_std,
    marker='o',
    label='Training error'
)

plt.errorbar(
    data_sizes,
    test_curve_mean,
    yerr=test_curve_std,
    marker='o',
    label='Test error'
)

plt.xlabel("Training set size")
plt.ylabel("Mean squared error")
plt.title("Learning curves for different data sizes")
plt.legend()
plt.grid(True)
plt.savefig("learning_curves_data_size.png", dpi=300)
plt.show()
