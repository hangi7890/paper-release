# Supplementary Formulas and Reference Code

## Window construction

Let `o` denote the first forecast-target timestamp. For building `b`, the input and direct 24-hour target are

`X[b,o] = [p(b,o-24), ..., p(b,o-1)]`

`y[b,o] = [p(b,o), ..., p(b,o+23)]`.

All 48 timestamps must be finite hourly observations. The 28 final origins are separated by 24 hours, so their target intervals do not overlap. For a nonzero adaptation budget `D`, a separate 24-hour input-history block plus `24D` target hours produces

`N(D) = 24(D+1) - 48 + 1 = 24D - 23`

hourly supervised windows. This yields 1, 49, 145, 313, and 649 windows for 1, 3, 7, 14, and 28 days. Zero-day adaptation has no target-building training label.

## Input representation and normalization

Past electricity is normalized with source-only statistics:

`z = (p - 167.0651976323119) / 354.328669571624`.

For target hour `t`, deterministic calendar features are

`hour_sin = sin(2π hour(t)/24)`, `hour_cos = cos(2π hour(t)/24)`,

`weekday_sin = sin(2π weekday(t)/7)`, `weekday_cos = cos(2π weekday(t)/7)`,

and `weekend = 1[weekday(t) >= 5]`.

The resulting C0 tensor is 24 × 6. Future measured electricity and weather are absent.

## Loss and adaptation

Both source training and target adaptation minimize mean squared error on source-normalized 24-hour targets:

`MSE = (1/(B×24)) Σ_i Σ_h (ŷ[i,h] - y[i,h])²`.

The completed source run used Adam with learning rate 0.001 and batch 256. Output-head adaptation used Adam with learning rate 0.0005 for 50 epochs. The actual loader batch is

`batch(D) = min(64, N(D))`,

with `shuffle=True` and `drop_last=False`. The LSTM parameters are frozen, and only the linear output-head parameters enter the adaptation optimizer.

## Error metrics

For the 672 target hours of a building-condition pair,

`MAE = (1/672) Σ_t |ŷ_t - y_t|`,

`RMSE = sqrt((1/672) Σ_t (ŷ_t - y_t)²)`.

Adaptation changes use `ΔMAE_source = MAE_adapt - MAE_source` and the analogous RMSE difference. Negative values indicate improvement relative to source-only.

## Site-equal aggregation

Seed-level effects are averaged within each building when required by the estimand. Buildings then receive equal weight within each site, and the seven site estimates receive equal weight in the overall estimate. The 95% interval resamples the seven sites with replacement 10,000 times using seed 20260925 and reports the 2.5th and 97.5th percentiles.

## Failure events

`EPS = 1e-12` is a floating-point tie tolerance, not a practical-effect threshold.

- F1: `ΔMAE_source > EPS`.
- F2: `ΔRMSE_source > EPS`.
- F3-PD: `MAE_adapt - MAE_previous_day > EPS`.
- F3-W: `MAE_adapt - MAE_weekly_same_hour > EPS`.
- F3-any: F3-PD or F3-W.
- F3-both: F3-PD and F3-W.
- F4: within one building-budget pair, at least one seed has `ΔMAE_source < -EPS` and at least one seed has `ΔMAE_source > EPS`.
- Metric discordance: `ΔMAE_source < -EPS` and `ΔRMSE_source > EPS`.

F1-F3 and discordance use building × budget × seed events. F4 uses building × budget events.

## Reference implementation

The executable, model-free implementation is `code/supplementary_reference_formulas.py`. It documents window counts, batches, MAE, RMSE, failure flags, site-equal means, and site-cluster percentile intervals. Running it does not train, adapt, or infer a model.
