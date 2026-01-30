"""
Project 10: Information-Theoretic Analysis of Representations

This experiment studies how mutual information between inputs and learned
representations changes as noise is introduced, illustrating information
loss and compression effects.
"""

import numpy as np
import matplotlib.pyplot as plt


# data generation

np.random.seed(42)

n_samples = 2000
X = np.random.normal(0, 1, n_samples)

# representation with controllable noise
noise_levels = np.linspace(0.01, 2.0, 20)


# entropy estimation (histogram-based)

def entropy(x, bins=50):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))


def joint_entropy(x, y, bins=50):
    hist, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))


def mutual_information(x, y):
    return entropy(x) + entropy(y) - joint_entropy(x, y)


# experiment

mi_values = []
rep_entropy = []

for sigma in noise_levels:
    Z = X + np.random.normal(0, sigma, n_samples)
    mi = mutual_information(X, Z)

    mi_values.append(mi)
    rep_entropy.append(entropy(Z))


# results

for s, mi in zip(noise_levels, mi_values):
    print(f"Noise std={round(s,2)} -> MI(X;Z)={round(mi,3)}")


# visualization

plt.figure(1)
plt.plot(noise_levels, mi_values, marker='o')
plt.xlabel("Noise standard deviation")
plt.ylabel("Mutual Information I(X; Z)")
plt.title("Information retained under noise")
plt.grid(True)
plt.savefig("mi_vs_noise.png", dpi=300)
plt.show()


plt.figure(2)
plt.plot(noise_levels, rep_entropy, marker='o')
plt.xlabel("Noise standard deviation")
plt.ylabel("Entropy of representation H(Z)")
plt.title("Representation entropy vs noise")
plt.grid(True)
plt.savefig("representation_entropy.png", dpi=300)
plt.show()

