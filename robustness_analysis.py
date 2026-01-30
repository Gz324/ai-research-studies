"""
Study: Robustness of classification models under distribution shift.

This experiment evaluates how predictive performance degrades when
input distributions are perturbed, simulating real-world noise and
shifted conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# data

np.random.seed(42)

X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    class_sep=1.5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# baseline model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, model.predict(X_test))


# distribution shift via noise

noise_levels = np.linspace(0, 1.0, 12)
accuracies = []

for sigma in noise_levels:
    noise = np.random.normal(0, sigma, X_test.shape)
    X_shifted = X_test + noise
    acc = accuracy_score(y_test, model.predict(X_shifted))
    accuracies.append(acc)


# results

print("Baseline accuracy:", round(baseline_acc, 3))
for s, a in zip(noise_levels, accuracies):
    print(f"Noise std={round(s,2)} -> accuracy={round(a,3)}")


# visualization

plt.figure()
plt.plot(noise_levels, accuracies, marker='o')
plt.xlabel("Noise standard deviation")
plt.ylabel("Accuracy")
plt.title("Model robustness under distribution shift")
plt.grid(True)
plt.savefig("robustness_curve.png", dpi=300)
plt.show()
