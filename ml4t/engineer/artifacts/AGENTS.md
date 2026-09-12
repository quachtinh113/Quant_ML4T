# artifacts/ - Lightweight Artifact Records

Helpers for representing feature, label, and prediction artifacts in reusable
pipeline workflows.

## Files

| File | Purpose |
|------|---------|
| `features.py` | feature artifact record helpers |
| `labels.py` | label artifact record helpers |
| `predictions.py` | prediction artifact record helpers |

## Notes

- this directory is a support surface, not the primary public API
- use it when orchestration code needs typed artifact records rather than raw frames
