"""
Study: Uncertainty quantification in probabilistic classification.

This experiment compares frequentist and Bayesian approaches by
analyzing predictive uncertainty and calibration behaviour.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# settings

N_SAMPLES = 3000
N_FEATURES = 10
MC_SAMPLES = 100


# data

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# frequentist model

freq_model = LogisticRegression(max_iter=1000)
freq_model.fit(X_train, y_train)

freq_probs = freq_model.predict_proba(X_test)[:, 1]


# bayesian-style approximation
# (Monte Carlo dropout idea via noise)

def noisy_predictions(model, X, noise_scale=0.1):
    coefs = model.coef_.copy()
    intercept = model.intercept_.copy()

    noisy_coefs = coefs + noise_scale * np.random.randn(*coefs.shape)
    logits = X @ noisy_coefs.T + intercept
    return 1 / (1 + np.exp(-logits))


mc_probs = []

for _ in range(MC_SAMPLES):
    mc_probs.append(noisy_predictions(freq_model, X_test))

mc_probs = np.array(mc_probs).squeeze()

bayes_mean = mc_probs.mean(axis=0)
bayes_std = mc_probs.std(axis=0)


# calibration analysis

bins = np.linspace(0, 1, 11)

def calibration_curve(probs, y_true):
    bin_acc = []
    bin_conf = []

    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            bin_acc.append(y_true[mask].mean())
            bin_conf.append(probs[mask].mean())

    return bin_conf, bin_acc


freq_conf, freq_acc = calibration_curve(freq_probs, y_test)
bayes_conf, bayes_acc = calibration_curve(bayes_mean, y_test)


# plots

plt.figure()
plt.plot(freq_conf, freq_acc, marker='o', label="Frequentist")
plt.plot(bayes_conf, bayes_acc, marker='o', label="Bayesian approx")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Predicted confidence")
plt.ylabel("Empirical accuracy")
plt.title("Calibration comparison")
plt.legend()
plt.grid(True)
plt.savefig("calibration_comparison.png", dpi=300)
plt.show()


plt.figure()
plt.hist(bayes_std, bins=30)
plt.xlabel("Predictive uncertainty (std)")
plt.ylabel("Frequency")
plt.title("Distribution of predictive uncertainty")
plt.grid(True)
plt.savefig("predictive_uncertainty.png", dpi=300)
plt.show()


# metrics

freq_brier = brier_score_loss(y_test, freq_probs)
bayes_brier = brier_score_loss(y_test, bayes_mean)

print("Brier score (frequentist):", round(freq_brier, 4))
print("Brier score (bayesian approx):", round(bayes_brier, 4))
