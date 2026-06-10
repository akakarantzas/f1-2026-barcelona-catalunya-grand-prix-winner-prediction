# F1 2026 Barcelona-Catalunya Grand Prix Winner Prediction

Standalone ML project for predicting the 2026 Barcelona-Catalunya Grand Prix winner.

The production app should not train this model at request time. This repository owns the data loading, feature engineering, training, validation, model artifact, and prediction export. The app can consume the exported `barcelona_catalunya_predictions.json` and, if needed later, the frozen `barcelona_catalunya_model.pkl`.

## Approach

- Loads historical Barcelona/Spanish GP results plus recent full-season form using FastF1.
- Builds rolling driver and team performance features.
- Trains a calibrated gradient boosting classifier with cross-validation.
- Exports a ranked win-probability JSON for the 2026 Barcelona-Catalunya GP.

## Run

```bash
pip install -r requirements.txt
python train_barcelona_catalunya.py
```

## Outputs

- `barcelona_catalunya_predictions.json`
- `barcelona_catalunya_model.pkl`
- `barcelona_catalunya_metadata.json`

## Production Integration

Copy only the exported JSON/model artifacts into `chicane-ai`. Keep notebooks, caches, and training code in this repository.

