"""
Study: Optimization dynamics and convergence behaviour.

This experiment compares gradient descent and stochastic gradient descent
in terms of convergence speed, stability, and optimization noise.
"""

import numpy as np
import matplotlib.pyplot as plt


# synthetic regression problem

np.random.seed(42)

N = 200
X = np.linspace(-2, 2, N)
y = 3 * X + 2 + np.random.normal(0, 0.5, N)

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


# loss and gradient

def mse_loss(w, X, y):
    preds = X @ w
    return np.mean((preds - y) ** 2)


def grad_mse(w, X, y):
    return 2 * X.T @ (X @ w - y) / len(y)


# optimization methods

def gradient_descent(X, y, lr=0.05, steps=120):
    w = np.zeros((1, 1))
    losses = []

    for _ in range(steps):
        w -= lr * grad_mse(w, X, y)
        losses.append(mse_loss(w, X, y))

    return w, losses


def stochastic_gradient_descent(X, y, lr=0.05, steps=120):
    w = np.zeros((1, 1))
    losses = []

    for _ in range(steps):
        i = np.random.randint(0, len(y))
        xi = X[i:i+1]
        yi = y[i:i+1]

        w -= lr * grad_mse(w, xi, yi)
        losses.append(mse_loss(w, X, y))

    return w, losses


# run experiments

w_gd, loss_gd = gradient_descent(X, y)
w_sgd, loss_sgd = stochastic_gradient_descent(X, y)


print("Final GD weight:", round(w_gd.item(), 3))
print("Final SGD weight:", round(w_sgd.item(), 3))
print("True weight: 3.0")


# visualization

plt.figure()
plt.plot(loss_gd, label="Gradient Descent")
plt.plot(loss_sgd, label="Stochastic Gradient Descent")
plt.xlabel("Optimization step")
plt.ylabel("MSE loss")
plt.title("Optimization convergence comparison")
plt.legend()
plt.grid(True)
plt.savefig("convergence_comparison.png", dpi=300)
plt.show()
