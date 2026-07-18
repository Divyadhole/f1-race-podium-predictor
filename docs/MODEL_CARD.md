# Model Card

## Model

The production artifact is a histogram gradient-boosting classifier selected on a 2023 validation period. It consumes seven numeric pre-race features and returns a podium probability for each driver.

## Evaluation

On 240 out-of-time rows from 2024-2025, the model reaches 0.2351 log loss, 0.0650 Brier score, 0.9520 ROC-AUC, 0.9188 PR-AUC, 92.50% accuracy, and 0.0241 calibration error.

## Comparison

Logistic regression performs better on several test metrics, but gradient boosting had the lower validation log loss and therefore remains the precommitted production choice. The grid-only benchmark is more poorly calibrated and less accurate.

## Risks

Synthetic relationships can overstate generalization. Independent driver probabilities do not enforce exactly three podium finishers. Dataset shift, regulations, constructor upgrades, and lineup changes require monitoring and retraining.

