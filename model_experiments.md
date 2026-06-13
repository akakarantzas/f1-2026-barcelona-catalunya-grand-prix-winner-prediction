# Model Experiments

Short record of model-history choices tested for the Barcelona-Catalunya winner model.

## Accepted Baseline

Data window:

- 2022-present Spanish GP and available full-season race data.
- No pre-2022 races.

Validation:

| Metric | Value |
| --- | ---: |
| Historical races loaded | 33 |
| Races tested | 25 |
| Top-1 accuracy | 0.60 |
| Top-3 accuracy | 1.00 |
| Top-5 accuracy | 1.00 |
| Mean winner rank | 1.48 |
| Log loss | 0.1013 |
| Brier score | 0.0307 |

Decision:

Keep this as the current baseline.

## Rejected: Legacy Barcelona Track History

Data window:

- 2022-present data.
- Added 2018-2021 Spanish GP only.
- Down-weighted legacy Spanish GP rows to `0.35`.

Validation:

| Metric | Value |
| --- | ---: |
| Historical races loaded | 37 |
| Races tested | 29 |
| Top-1 accuracy | 0.5172 |
| Top-3 accuracy | 1.00 |
| Top-5 accuracy | 1.00 |
| Mean winner rank | 1.66 |
| Log loss | 0.1028 |
| Brier score | 0.0310 |

Decision:

Reject. More Barcelona history added samples, but winner ranking got worse. The older regulation-era races are not worth including in the current model.

## Accepted: Barcelona Feature Ablation (v1.4)

Method:

- Walk-forward ablation on the selected postprocess config.
- Compare full model vs dropping `BarcelonaExperience`/`BarcelonaWinRate`, zeroing the track prior, and removing both signals.
- Report global metrics plus Spain-only and non-Spain slices.

Baseline (full model):

| Slice | Races tested | Top-1 | Mean winner rank | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 25 | 0.60 | 1.48 | 0.1013 | 0.0307 |
| Spain | 1 | 1.00 | 1.00 | 0.1037 | 0.0313 |
| Non-Spain | 24 | 0.5833 | 1.50 | 0.1012 | 0.0307 |

Drop Barcelona model features:

| Slice | Top-1 | Mean winner rank | Log loss | Brier | Delta top-1 vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 0.40 | 1.72 | 0.1060 | 0.0322 | -0.20 |
| Spain | 0.00 | 2.00 | 0.1179 | 0.0368 | -1.00 |
| Non-Spain | 0.4167 | 1.71 | 0.1056 | 0.0320 | -0.1666 |

Zero Barcelona track prior:

| Slice | Top-1 | Mean winner rank | Log loss | Brier | Delta top-1 vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 0.56 | 1.52 | 0.0968 | 0.0297 | -0.04 |
| Spain | 1.00 | 1.00 | 0.0956 | 0.0287 | 0.00 |
| Non-Spain | 0.5417 | 1.54 | 0.0969 | 0.0298 | -0.0416 |

Drop Barcelona features and track prior:

| Slice | Top-1 | Mean winner rank | Log loss | Brier | Delta top-1 vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 0.48 | 1.64 | 0.1016 | 0.0312 | -0.12 |
| Spain | 1.00 | 1.00 | 0.1109 | 0.0346 | 0.00 |
| Non-Spain | 0.4583 | 1.67 | 0.1012 | 0.0311 | -0.125 |

Decision:

Keep Barcelona model features and the tuned track prior. Removing `BarcelonaExperience` and `BarcelonaWinRate` hurt both global and Spain-only winner ranking the most. The track prior adds a smaller but still positive top-1 contribution on the full backtest.

Note:

Only one Spain race (2025 Spanish GP) is in the walk-forward test window because earlier Spanish GPs are used as training history. Treat Spain-only deltas as directional, not statistically stable on their own.
