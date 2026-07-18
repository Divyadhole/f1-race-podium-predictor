# Engineering Decisions

## Chronological Rather Than Random Splits

Random splits would allow later form and constructor performance to influence earlier examples. Calendar boundaries match real deployment, where the future is unknown.

## Grid Baseline

Qualifying position is the strongest readily available signal. A model that cannot improve accuracy, log loss, and calibration beyond a grid-based prior adds little value.

## Shift Every Rolling Feature

Even a correctly sorted rolling window leaks the current race unless shifted. Every result-derived feature is shifted before aggregation.

## Validation-Based Model Selection

The test set is not a tuning tool. Gradient boosting is retained because it won the 2023 selection period, even though logistic regression later performed better on the test period.

## Synthetic Offline Tables

The project remains runnable without Kaggle credentials. Public tables can replace the sample because schemas and relational checks are explicit.

