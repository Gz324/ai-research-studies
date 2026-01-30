# Stability and Faithfulness of Model Explanations

This project studies whether feature importance explanations produced by
machine learning models are reliable and faithful to the model’s actual
decision behaviour.

## Overview
Rather than focusing on generating explanations, the experiments evaluate
their stability across multiple runs and their faithfulness through controlled
feature-removal tests.

## Experiments
- Stability of permutation-based feature importance across random initializations
- Rank agreement of explanations between different model classes
- Faithfulness testing by removing top-ranked features and observing performance degradation

## Models
- Logistic Regression
- Random Forest

## Files
- explanation_stability.py: experiment code
- importance_stability.png: stability analysis across runs
- faithfulness_test.png: faithfulness evaluation via feature removal
