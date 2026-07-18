# Architecture

## Data Boundary

`data.py` owns table schemas, primary keys, foreign keys, duplicate detection, and offline generation. `features.py` owns joins and all target-aware transformations. Joining uses pandas cardinality validation so an accidental many-to-many expansion fails immediately.

## Temporal Safety

Rows are ordered by race date. Driver and constructor rolling metrics call `shift()` before `rolling()`, and circuit history calls `shift()` before `expanding()`. Training ends in 2023; the test period begins in 2024.

## Model Selection

The last training year is held out internally. Four gradient configurations and a logistic model are compared there. Only the winner is fitted on all pre-2024 rows and serialized. The 2024-2025 test period is used once for reporting.

## Serving

The artifact contains the fitted preprocessing/model pipeline. CLI and FastAPI share `PodiumModel.predict_one`. The API validates feature ranges and supports both single-driver and full-grid ranking requests.

