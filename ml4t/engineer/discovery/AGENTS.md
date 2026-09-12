# discovery/ - Feature Discovery Surface

Metadata-driven search and filtering for the feature registry.

## Files

| File | Purpose | Key Objects |
|------|---------|-------------|
| `catalog.py` | catalog wrapper around registry metadata | `FeatureCatalog`, `features` |

## Typical Usage

```python
from ml4t.engineer import feature_catalog

feature_catalog.list(category="momentum")
feature_catalog.search("volatility estimator")
feature_catalog.describe("rsi")
```
