"""
Study: Measuring algorithmic fairness in binary classification.

This experiment analyzes group fairness by comparing prediction outcomes
across sensitive attribute groups and examining trade-offs between accuracy
and fairness.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# data generation

np.random.seed(42)

X, y = make_classification(
    n_samples=4000,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    weights=[0.6, 0.4],
    random_state=42
)

# sensitive attribute (e.g., group membership)
sensitive = (X[:, 0] > X[:, 0].mean()).astype(int)

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)


# model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# fairness metrics

def group_rate(preds, group):
    return preds[group == 1].mean(), preds[group == 0].mean()


pos_rate_1, pos_rate_0 = group_rate(y_pred, s_test)
demographic_parity_gap = abs(pos_rate_1 - pos_rate_0)


def true_positive_rate(y_true, y_pred, group):
    mask = (group == 1) & (y_true == 1)
    if mask.sum() == 0:
        return 0
    return y_pred[mask].mean()


tpr_1 = true_positive_rate(y_test, y_pred, s_test)
tpr_0 = true_positive_rate(y_test, y_pred, 1 - s_test)
equal_opportunity_gap = abs(tpr_1 - tpr_0)


accuracy = accuracy_score(y_test, y_pred)


# results

print("Accuracy:", round(accuracy, 3))
print("Demographic parity gap:", round(demographic_parity_gap, 3))
print("Equal opportunity gap:", round(equal_opportunity_gap, 3))


# visualization

plt.figure(1)
plt.bar(["Group 0", "Group 1"], [pos_rate_0, pos_rate_1])
plt.ylabel("Positive prediction rate")
plt.title("Demographic parity analysis")
plt.grid(True)
plt.savefig("demographic_parity.png", dpi=300)
plt.show()


plt.figure(2)
plt.bar(["Group 0", "Group 1"], [tpr_0, tpr_1])
plt.ylabel("True positive rate")
plt.title("Equal opportunity analysis")
plt.grid(True)
plt.savefig("equal_opportunity.png", dpi=300)
plt.show()
