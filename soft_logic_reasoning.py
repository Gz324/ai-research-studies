"""
Study: Reasoning under soft logical constraints.

This experiment studies how conflicting logical rules can be handled
using soft constraints, where violations are penalized rather than forbidden.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt


# Logical variables
# A: has symptoms
# B: test positive
# C: disease present

variables = ["A", "B", "C"]


# Soft logical rules (penalties)
# Each rule returns 0 if satisfied, positive penalty if violated

def rule_symptoms_imply_disease(state):
    # A -> C
    if state["A"] and not state["C"]:
        return 2
    return 0


def rule_test_implies_disease(state):
    # B -> C
    if state["B"] and not state["C"]:
        return 3
    return 0


def rule_no_symptoms_means_no_disease(state):
    # not A -> not C
    if not state["A"] and state["C"]:
        return 1
    return 0


rules = [
    rule_symptoms_imply_disease,
    rule_test_implies_disease,
    rule_no_symptoms_means_no_disease
]


# Generate all possible worlds

def all_assignments(vars):
    for values in itertools.product([False, True], repeat=len(vars)):
        yield dict(zip(vars, values))


states = list(all_assignments(variables))


# Evaluate worlds

def total_penalty(state):
    return sum(rule(state) for rule in rules)


penalties = []
labels = []

for s in states:
    penalties.append(total_penalty(s))
    label = "".join([v if s[v] else f"¬{v}" for v in variables])
    labels.append(label)


# Analysis

penalties = np.array(penalties)
best_states = penalties == penalties.min()


print("Minimum penalty:", penalties.min())
print("Best satisfying worlds:")
for i, is_best in enumerate(best_states):
    if is_best:
        print(labels[i])


# Visualization

plt.figure(figsize=(10, 4))
plt.bar(labels, penalties)
plt.axhline(penalties.min(), linestyle="--", label="Minimum penalty")
plt.ylabel("Total penalty")
plt.title("Soft logical constraint satisfaction")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("soft_logic_penalties.png", dpi=300)
plt.show()
