"""
Study: Counterfactual reasoning in a simple causal model.

This experiment demonstrates how predictions change under interventions
and counterfactual scenarios, highlighting the difference between
correlation-based prediction and causal reasoning.
"""

import numpy as np
import matplotlib.pyplot as plt


# Causal model
# X: treatment (e.g., medication)
# Z: confounder (e.g., severity)
# Y: outcome (e.g., recovery score)
#
# Z -> X
# Z -> Y
# X -> Y


np.random.seed(42)
N = 2000

Z = np.random.normal(0, 1, N)                 # confounder
X = (Z + np.random.normal(0, 0.5, N)) > 0     # treatment assignment
X = X.astype(float)

Y = 2 * X + Z + np.random.normal(0, 0.5, N)   # outcome


# Observational estimation (correlation-based)

beta_obs = np.polyfit(X, Y, 1)[0]


# Interventional estimation (do(X))

def intervene_do_x(x_value, z):
    return 2 * x_value + z + np.random.normal(0, 0.5, len(z))


Y_do_0 = intervene_do_x(0, Z)
Y_do_1 = intervene_do_x(1, Z)

ate = Y_do_1.mean() - Y_do_0.mean()


# Counterfactual reasoning
# For individuals who received X=0, ask:
# "What would Y have been if X were 1?"

mask = X == 0
Z_cf = Z[mask]

Y_factual = Y[mask]
Y_counterfactual = intervene_do_x(1, Z_cf)

cf_effect = Y_counterfactual.mean() - Y_factual.mean()


# Results

print("Observational effect (correlation):", round(beta_obs, 3))
print("Average Treatment Effect (do-operator):", round(ate, 3))
print("Counterfactual effect (X=0 -> X=1):", round(cf_effect, 3))


# Visualization

plt.figure(1)
plt.scatter(X + np.random.normal(0, 0.02, N), Y, alpha=0.3)
plt.xlabel("Treatment (X)")
plt.ylabel("Outcome (Y)")
plt.title("Observed data (confounded)")
plt.grid(True)
plt.savefig("observational_data.png", dpi=300)
plt.show()


plt.figure(2)
plt.bar(["do(X=0)", "do(X=1)"], [Y_do_0.mean(), Y_do_1.mean()])
plt.ylabel("Expected outcome")
plt.title("Interventional comparison")
plt.grid(True)
plt.savefig("interventional_effect.png", dpi=300)
plt.show()
