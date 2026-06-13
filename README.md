# F1 2026 Barcelona-Catalunya Grand Prix Winner Prediction

Standalone ML project for predicting the 2026 Barcelona-Catalunya Grand Prix winner.

This repository owns data loading, feature engineering, validation, training, and prediction export. The production app consumes exported artifacts; it does not train the model at request time.

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
- Trains a calibrated `HistGradientBoostingClassifier`.
- Uses walk-forward backtesting to tune the validated probability blend:
  - model probability
  - recent form prior
  - grid-position prior
  - Barcelona-track prior
- Applies current-race prediction-only priors after validation:
  - recent dominance
  - optional market odds
  - winner-contender allocation
- Exports ranked win probabilities and metadata for the app.

Prediction-only priors affect the exported 2026 Barcelona-Catalunya prediction, but they are not counted as historical validation improvements.

## Current Model

- Version: `barcelona-catalunya-hgb-calibrated-1.4`
- Training samples: `671`
- Historical races loaded: `33`

Walk-forward validation:

Validation starts after the first 8 loaded races are available as training history, so 25 of the 33 loaded races are tested.

| Metric | Value |
| --- | ---: |
| Races tested | 25 |
| Top-1 accuracy | 0.60 |
| Top-3 accuracy | 1.00 |
| Top-5 accuracy | 1.00 |
| Mean winner rank | 1.48 |
| Log loss | 0.1013 |
| Brier score | 0.0307 |

v1.4 also exports `barcelona_significance` ablation results and Spain vs non-Spain validation slices in metadata. See `model_experiments.md` for accepted and rejected experiments.

Prediction-only winner allocation:

- Contenders: `ANT`, `NOR`, `PIA`, `RUS`, `VER`, `LEC`, `HAM`
- Contender probability mass: `0.93`
- Rest-of-grid tail probability mass: `0.07`

The contender allocation affects only the exported race prediction. It is not included in walk-forward validation metrics.

Current exported top five (pre-qualifying, with full market odds prior):

| Rank | Driver | Team | Probability |
| ---: | --- | --- | ---: |
| 1 | Antonelli | Mercedes | 0.2825 |
| 2 | Norris | McLaren | 0.2511 |
| 3 | Piastri | McLaren | 0.1720 |
| 4 | Russell | Mercedes | 0.1281 |
| 5 | Verstappen | Red Bull Racing | 0.0499 |

## Run

```bash
pip install -r requirements.txt
python train_barcelona_catalunya.py
```

The script writes the exported artifacts in the repository root.

## Prediction Inputs

The exporter can run from defaults in `train_barcelona_catalunya.py`. Optional JSON files override or extend prediction inputs:

- `qualifying_grid.json`: actual grid positions after qualifying.
- `market_odds.json`: pre-race market odds used as a prediction-only prior.

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

For cleaner probabilities, include odds for all realistic contenders, not only one driver. A single-driver odds file assigns the full market prior to that driver after normalization.

## Outputs

- `barcelona_catalunya_predictions.json`
- `barcelona_catalunya_metadata.json`
- `barcelona_catalunya_model.pkl`

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
