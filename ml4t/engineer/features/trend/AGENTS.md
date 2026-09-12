# trend/ - 10 Trend Indicators

Moving-average and price-level trend features.

## Current Modules

| Module | Purpose |
|--------|---------|
| `sma.py` | simple moving average |
| `ema.py` | exponential moving average |
| `wma.py` | weighted moving average |
| `dema.py` | double EMA |
| `tema.py` | triple EMA |
| `t3.py` | Tillson T3 smoothing |
| `kama.py` | Kaufman adaptive moving average |
| `trima.py` | triangular moving average |
| `midpoint.py` | midpoint over a rolling window |
| `donchian.py` | Donchian channel boundaries |

## Usage Notes

- these features are price-scale outputs, so most models will need preprocessing
- for trend strength rather than level-following outputs, combine this directory with
  momentum indicators such as `adx` or `aroon`
- see `docs/user-guide/features.md` and `docs/user-guide/ml-readiness.md` for the
  production-facing guidance
