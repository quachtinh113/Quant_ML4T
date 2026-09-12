# volatility/ - 15 Volatility Measures

Range-based, realized, and conditional volatility estimators.

## Core Modules

| Module | Purpose |
|--------|---------|
| `atr.py` | average true range |
| `natr.py` | normalized ATR |
| `trange.py` | single-bar true range |
| `bollinger_bands.py` | Bollinger bands and bandwidth |
| `yang_zhang_volatility.py` | Yang-Zhang estimator |
| `parkinson_volatility.py` | Parkinson high-low estimator |
| `garman_klass_volatility.py` | Garman-Klass OHLC estimator |
| `rogers_satchell_volatility.py` | drift-robust Rogers-Satchell estimator |
| `realized_volatility.py` | realized volatility |
| `ewma_volatility.py` | EWMA volatility |
| `garch_forecast.py` | GARCH forecast |
| `conditional_volatility_ratio.py` | short/long volatility ratio |
| `volatility_percentile_rank.py` | current volatility percentile |
| `volatility_of_volatility.py` | second-order volatility |
| `volatility_regime_probability.py` | high/low-volatility regime probability |

## Practical Ranking

- baseline: `atr`, `natr`, `bollinger_bands`
- research-grade estimators: `yang_zhang_volatility`, `garman_klass_volatility`,
  `rogers_satchell_volatility`
- regime/context features: `conditional_volatility_ratio`,
  `volatility_percentile_rank`, `volatility_regime_probability`
