# Data Card

## Dataset

The repository includes six deterministic CSV tables: races, circuits, drivers, constructors, qualifying, and results. They contain 108 races, ten drivers, five constructors, twelve circuits, and 1,080 driver-race outcomes from 2017 through 2025.

## Provenance

The tables are generated under seed 42 to mirror the relational shape of the recommended Kaggle/Ergast Formula 1 dataset. Latent driver and constructor strength, season trends, grid noise, circuit effects, race noise, and reliability create realistic relationships without presenting fictional rows as historical fact.

## Quality Rules

- Unique race, driver, and constructor keys
- Unique driver-race rows in qualifying and results
- Valid foreign keys across all result tables
- Positive grid and finish positions
- Parseable race dates
- Complete one-to-one qualifying/result joins

## Appropriate Use

Use for portfolio demonstration, software testing, and methodological experiments. Do not use predictions trained on this data for betting, team decisions, or claims about real drivers.

