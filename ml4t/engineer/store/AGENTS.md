# store/ - Offline Feature Store

DuckDB-based offline storage helpers for engineered features and related artifacts.

## Files

| File | Purpose | Key Export |
|------|---------|------------|
| `offline.py` | local feature-store implementation | `OfflineFeatureStore` |

## Notes

- lower-priority than the main compute, labeling, and dataset workflows
- useful when feature generation needs persistence between runs
