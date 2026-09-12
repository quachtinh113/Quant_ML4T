# core/ - Registry, Metadata, and Validation

The internal foundation of `ml4t-engineer`.

## Files

| File | Purpose |
|------|---------|
| `registry.py` | feature metadata registry and lookup helpers |
| `decorators.py` | `@feature` registration decorator |
| `schemas.py` | canonical schemas and column expectations |
| `types.py` | shared type aliases and protocol-like definitions |
| `validation.py` | runtime validation helpers |
| `exceptions.py` | custom exception types |

## Notes

- this directory underpins `compute_features()` and `feature_catalog`
- it is mostly an implementation surface, but useful when debugging registration,
  lookback metadata, or input validation

See [calendars/AGENTS.md](calendars/AGENTS.md) for the nested trading-calendar helpers.
