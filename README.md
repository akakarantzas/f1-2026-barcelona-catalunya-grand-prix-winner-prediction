# F1 2026 Barcelona-Catalunya Grand Prix Winner Prediction

Standalone ML project for predicting the 2026 Barcelona-Catalunya Grand Prix winner.

The production app should not train this model at request time. This repository owns the data loading, feature engineering, training, validation, model artifact, and prediction export. The app can consume the exported `barcelona_catalunya_predictions.json` and, if needed later, the frozen `barcelona_catalunya_model.pkl`.

## Tech Stack

- Python
- FastF1 for historical race data
- pandas and NumPy for feature engineering
- scikit-learn for calibrated gradient boosting
- joblib for model serialization
- JSON artifacts for app integration

## Approach

- Loads historical Barcelona/Spanish GP results plus recent full-season form using FastF1.
- Builds rolling driver and team performance features.
- Trains a calibrated gradient boosting classifier with cross-validation.
- Tunes post-processing with walk-forward backtesting.
- Blends validated model probabilities with form, grid, Barcelona-track, recent-dominance, and optional market-odds priors.
- Exports ranked win-probability JSON and metadata for the 2026 Barcelona-Catalunya GP.

## Current Model

- Version: `barcelona-catalunya-hgb-calibrated-1.3`
- Training samples: `671`
- Training races loaded: `33`

Walk-forward validation:

| Metric | Value |
| --- | ---: |
| Races tested | 25 |
| Top-1 accuracy | 0.60 |
| Top-3 accuracy | 1.00 |
| Top-5 accuracy | 1.00 |
| Mean winner rank | 1.48 |
| Log loss | 0.1013 |
| Brier score | 0.0307 |

## Run

```bash
pip install -r requirements.txt
python train_barcelona_catalunya.py
```

The script writes the exported artifacts in the repository root.

## Prediction Inputs

The exporter can run with only the defaults in `train_barcelona_catalunya.py`, but it also supports optional JSON inputs:

- `qualifying_grid.json`: actual or overridden grid positions after qualifying.
- `market_odds.json`: pre-race market odds used as a small prediction-only prior.

Example files:

- `qualifying_grid.example.json`
- `market_odds.example.json`

## Qualifying Grid

Before qualifying, the script uses the projected grid in `train_barcelona_catalunya.py`. After qualifying, copy `qualifying_grid.example.json` to `qualifying_grid.json`, update the positions, and rerun training/export.

## Market Odds

`market_odds.json` is optional. When present, the script validates driver codes, converts decimal odds to implied probabilities, normalizes them into fair market probabilities, and records the source metadata in `barcelona_catalunya_metadata.json`.

For each driver:

```text
implied_probability = 1 / decimal_odds
fair_probability = implied_probability / sum(all_implied_probabilities)
```

The market prior is prediction-only. It does not change the walk-forward validation metrics because those metrics remain based on historical races.

## Outputs

- `barcelona_catalunya_predictions.json`
- `barcelona_catalunya_model.pkl`
- `barcelona_catalunya_metadata.json`

## Production Integration

`chicane-ai` consumes the exported prediction artifacts:

- `barcelona_catalunya_predictions.json`
- `barcelona_catalunya_metadata.json`
- `barcelona_catalunya_model.pkl` when explicitly synced

From the `chicane-ai` repository, sync artifacts with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_barcelona_model.ps1 -SourceRepo ..\f1-2026-barcelona-catalunya-grand-prix-winner-prediction
```

Include the frozen model pickle only when the trained estimator changed:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_barcelona_model.ps1 -SourceRepo ..\f1-2026-barcelona-catalunya-grand-prix-winner-prediction -IncludeModel
```

Training code and FastF1 cache data stay in this repository.
