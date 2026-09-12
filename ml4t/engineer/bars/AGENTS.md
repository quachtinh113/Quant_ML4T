# bars/ - Alternative Bar Sampling

Information-driven bar samplers for converting trade-level data into OHLCV bars that
track market activity instead of clock time.

## Public Samplers

### Standard event-driven bars

- `TickBarSampler`
- `VolumeBarSampler`
- `DollarBarSampler`

### Adaptive or information-driven bars

- `ImbalanceBarSampler` for volume-imbalance bars
- `TickImbalanceBarSampler`
- `FixedTickImbalanceBarSampler`
- `FixedVolumeImbalanceBarSampler`
- `WindowTickImbalanceBarSampler`
- `WindowVolumeImbalanceBarSampler`

### Run bars

- `TickRunBarSampler`
- `VolumeRunBarSampler`
- `DollarRunBarSampler`

## Recommended Usage

- production: fixed-threshold imbalance bars for stable behavior
- research: window-based or adaptive imbalance bars when threshold adaptation matters
- baseline comparisons: tick, volume, and dollar bars

## Input Contract

All samplers expect trade-level data with at least:

- `timestamp`
- `price`
- `volume`

Imbalance and run bars also require side information or a trade-classification path
that the sampler can derive.

## Files Reference

| File | Purpose |
|------|---------|
| `base.py` | common sampler interface |
| `vectorized.py` | vectorized default implementations |
| `tick.py` | tick-bar implementations |
| `volume.py` | volume and dollar bars |
| `imbalance.py` | imbalance and fixed-threshold samplers |
| `run.py` | run-bar implementations |

## Example

```python
from ml4t.engineer.bars import (
    VolumeBarSampler,
    FixedTickImbalanceBarSampler,
)

volume_bars = VolumeBarSampler(volume_per_bar=50_000).sample(trades_df)
info_bars = FixedTickImbalanceBarSampler(threshold=100).sample(trades_df)
```

## Reference

See `docs/user-guide/bars.md` for the current production guidance and the tradeoffs
between adaptive, fixed-threshold, and window-based approaches.
