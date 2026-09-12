# statistics/ - Statistical and Regression Features

Rolling statistics, regression features, and structural-break helpers.

## Core Modules

| Module | Purpose |
|--------|---------|
| `stddev.py` | rolling standard deviation |
| `var.py` | rolling variance |
| `avgdev.py` | average deviation |
| `linearreg.py` | regression value |
| `linearreg_slope.py` | regression slope |
| `linearreg_angle.py` | regression angle |
| `linearreg_intercept.py` | regression intercept |
| `tsf.py` | time-series forecast |
| `structural_break.py` | structural-break and instability diagnostics |

## Notes

- the TA-Lib-compatible regression family lives here
- `structural_break.py` is heavier than the classic rolling-stat modules and is best
  treated as a specialized workflow
- additional normalized statistical features are surfaced through the registry and the
  user-guide metadata views
