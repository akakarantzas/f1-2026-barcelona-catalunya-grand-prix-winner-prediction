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
