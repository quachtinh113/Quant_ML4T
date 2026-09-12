# microstructure/ - 15 Market Microstructure Features

Microstructure features for liquidity, spread estimation, order flow, and trade
intensity. These features are most useful when the research question depends on
market quality rather than just price direction.

## Current Modules

| Module | Purpose |
|--------|---------|
| `kyle_lambda.py` | Kyle price-impact coefficient |
| `amihud_illiquidity.py` | illiquidity from return per dollar volume |
| `roll_spread_estimator.py` | spread estimation from return covariance |
| `realized_spread.py` | realized spread and reversal-based cost proxy |
| `effective_tick_rule.py` | trade-sign inference from price moves |
| `order_flow_imbalance.py` | buy/sell pressure imbalance |
| `price_impact_ratio.py` | impact relative to traded size |
| `quote_stuffing_indicator.py` | quote stuffing proxy |
| `trade_intensity.py` | trade arrival intensity |
| `volume_at_price_ratio.py` | volume concentration by price level |
| `volume_synchronicity.py` | volume comovement |
| `volume_weighted_price_momentum.py` | VWAP-style momentum measure |
| `order_book.py` | bid/ask imbalance, depth ratio, weighted mid-price |

## Practical Anchors

- liquidity proxies: `kyle_lambda`, `amihud_illiquidity`
- spread proxies: `roll_spread_estimator`, `realized_spread`
- order-flow signals: `order_flow_imbalance`, `effective_tick_rule`
- quote and book-state signals: `order_book.py`, `quote_stuffing_indicator`

## Notes

- some signals assume richer quote or trade data than plain OHLCV
- the NASDAQ-100 microstructure case study overlaps with these implementations, but
  often builds them manually for pedagogy
- see `docs/user-guide/features.md` for the production-facing feature overview
