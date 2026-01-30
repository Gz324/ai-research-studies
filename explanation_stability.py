"""
Study: Stability and faithfulness of model explanations.

This experiment investigates whether feature importance explanations
are stable across runs and whether they are faithful to the model's
actual decision behaviour.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau


# basic settings

RUNS = 5
N_FEATURES = 10
RANDOM_STATE = 42


# data

X, y = make_classification(
    n_samples=2000,
    n_features=N_FEATURES,
    n_informative=5,
    n_redundant=2,
    random_state=RANDOM_STATE
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
}


# permutation importance

def permutation_importance(model, X, y):
    baseline = accuracy_score(y, model.predict(X))
    importances = []

    for j in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, j])
        score = accuracy_score(y, model.predict(X_permuted))
        importances.append(baseline - score)

    return np.array(importances)


# Experiment 1: Stability across runs

importance_means = {}
importance_stds = {}

for name, model in models.items():
    all_importances = []

    for seed in range(RUNS):
        np.random.seed(seed)
        model.fit(X_train, y_train)
        imp = permutation_importance(model, X_test, y_test)
        all_importances.append(imp)

    all_importances = np.array(all_importances)
    importance_means[name] = np.mean(all_importances, axis=0)
    importance_stds[name] = np.std(all_importances, axis=0)


plt.figure()
for name in models:
    plt.errorbar(
        range(N_FEATURES),
        importance_means[name],
        yerr=importance_stds[name],
        marker='o',
        label=name
    )

plt.xlabel("Feature index")
plt.ylabel("Importance (accuracy drop)")
plt.title("Stability of feature importance across runs")
plt.legend()
plt.grid(True)
plt.savefig("importance_stability.png", dpi=300)
plt.show()


# Experiment 2: Rank agreement between models

lr_rank = np.argsort(-importance_means["Logistic Regression"])
rf_rank = np.argsort(-importance_means["Random Forest"])

tau, _ = kendalltau(lr_rank, rf_rank)

print("Kendall tau rank agreement between models:", round(tau, 3))


# Experiment 3: Faithfulness via feature removal

k_values = [1, 3, 5]
faithfulness_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    base_accuracy = accuracy_score(y_test, model.predict(X_test))
    ranked_features = np.argsort(-importance_means[name])

    drops = []

    for k in k_values:
        X_modified = X_test.copy()
        X_modified[:, ranked_features[:k]] = 0
        acc = accuracy_score(y_test, model.predict(X_modified))
        drops.append(base_accuracy - acc)

    faithfulness_results[name] = drops


plt.figure()
for name in faithfulness_results:
    plt.plot(
        k_values,
        faithfulness_results[name],
        marker='o',
        label=name
    )

plt.xlabel("Top-k features removed")
plt.ylabel("Accuracy drop")
plt.title("Faithfulness of explanations via feature removal")
plt.legend()
plt.grid(True)
plt.savefig("faithfulness_test.png", dpi=300)
plt.show()
