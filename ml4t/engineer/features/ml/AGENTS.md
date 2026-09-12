# ml/ - 14 ML-Oriented Feature Utilities

Utilities intended for model-ready transforms rather than classical technical
analysis indicators.

## Current Modules

| Module | Purpose |
|--------|---------|
| `create_lag_features.py` | lagged versions of one or more series |
| `cyclical_encode.py` | cyclic encodings such as hour/day sin-cos transforms |
| `directional_targets.py` | directional targets and simple label helpers |
| `fourier_features.py` | Fourier-based seasonality features |
| `interaction_features.py` | pairwise or structured feature interactions |
| `multi_horizon_returns.py` | returns over multiple future or backward horizons |
| `percentile_rank_features.py` | rank-based normalization and percentile context |
| `regime_conditional_features.py` | transforms conditioned on regime inputs |
| `rolling_entropy.py` | entropy-style complexity measures |
| `time_decay_weights.py` | recency weighting for observations |
| `volatility_adjusted_returns.py` | return normalization by volatility |

## Related Standalone Helpers

- `features/fdiff.py` for fractional differencing
- `dataset.py` for leakage-safe assembly into model-ready datasets
- `preprocessing.py` for train-only scaling and transform pipelines

## Usage Pattern

These utilities are useful when raw indicators are not enough and you need:

- lags and horizon structure
- rank or entropy-based transforms
- cyclical calendar encodings
- model-oriented interaction terms
