"""
Project 9: Representation Learning with Autoencoders

This experiment studies how an autoencoder learns low-dimensional
representations of data and how compression affects reconstruction quality.
"""

import numpy as np
import matplotlib.pyplot as plt


# synthetic dataset (2D structure)

np.random.seed(42)

n_samples = 500
theta = np.linspace(0, 2 * np.pi, n_samples)
x1 = np.cos(theta) + np.random.normal(0, 0.05, n_samples)
x2 = np.sin(theta) + np.random.normal(0, 0.05, n_samples)

X = np.vstack([x1, x2]).T


# autoencoder parameters

input_dim = 2
latent_dim = 1

W_enc = np.random.randn(input_dim, latent_dim) * 0.1
W_dec = np.random.randn(latent_dim, input_dim) * 0.1

lr = 0.05
epochs = 800


# helper functions

def encode(X):
    return X @ W_enc


def decode(Z):
    return Z @ W_dec


def reconstruction_loss(X, X_hat):
    return np.mean((X - X_hat) ** 2)


# training loop

losses = []

for _ in range(epochs):
    Z = encode(X)
    X_hat = decode(Z)

    loss = reconstruction_loss(X, X_hat)
    losses.append(loss)

    grad_Xhat = 2 * (X_hat - X) / len(X)
    grad_W_dec = Z.T @ grad_Xhat
    grad_Z = grad_Xhat @ W_dec.T
    grad_W_enc = X.T @ grad_Z

    W_dec -= lr * grad_W_dec
    W_enc -= lr * grad_W_enc


# final representations

Z_final = encode(X)
X_reconstructed = decode(Z_final)

print("Final reconstruction error:", round(losses[-1], 5))


# visualization

# reconstruction error curve
plt.figure(1)
plt.plot(losses)
plt.xlabel("Training epoch")
plt.ylabel("Reconstruction error")
plt.title("Autoencoder training convergence")
plt.grid(True)
plt.savefig("reconstruction_error.png", dpi=300)
plt.show()

# latent space
plt.figure(2)
plt.scatter(Z_final, np.zeros_like(Z_final), s=10)
plt.xlabel("Latent dimension")
plt.title("Learned 1D latent representation")
plt.yticks([])
plt.grid(True)
plt.savefig("latent_space.png", dpi=300)
plt.show()

