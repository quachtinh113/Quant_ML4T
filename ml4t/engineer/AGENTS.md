# ml4t.engineer Package

Package-level navigation for the public `ml4t-engineer` surface.

## Main Modules

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `api.py` | Feature-computation entry point | `compute_features()` |
| `dataset.py` | Leakage-safe dataset preparation | `MLDatasetBuilder`, `create_dataset_builder()` |
| `preprocessing.py` | Train-only scalers and transform pipelines | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `PreprocessingPipeline` |
| `discovery/catalog.py` | Metadata-driven feature exploration | `FeatureCatalog`, `feature_catalog` |
| `config/` | Reusable config models | `LabelingConfig`, `PreprocessingConfig`, `DataContractConfig` |
| `__init__.py` | Public re-exports and AGENTS discovery helper | `get_agent_docs()` |

## Subdirectories

| Directory | Purpose | AGENTS |
|-----------|---------|--------|
| `features/` | Registry-backed indicator implementations and standalone feature helpers | [features/AGENTS.md](features/AGENTS.md) |
| `labeling/` | Barrier labels, percentile labels, meta-labeling, uniqueness | [labeling/AGENTS.md](labeling/AGENTS.md) |
| `bars/` | Tick, volume, dollar, imbalance, and run bars | [bars/AGENTS.md](bars/AGENTS.md) |
| `core/` | Registry, metadata, schemas, decorators, validation | [core/AGENTS.md](core/AGENTS.md) |
| `config/` | Pydantic configuration models and schema bridges | [config/AGENTS.md](config/AGENTS.md) |
| `discovery/` | Metadata-driven feature search and filtering | [discovery/AGENTS.md](discovery/AGENTS.md) |
| `relationships/` | Correlation helpers and plotting utilities | [relationships/AGENTS.md](relationships/AGENTS.md) |
| `store/` | Offline DuckDB storage helpers | [store/AGENTS.md](store/AGENTS.md) |
| `artifacts/` | Lightweight artifact records for features, labels, predictions | [artifacts/AGENTS.md](artifacts/AGENTS.md) |
| `logging/` | Structured logging configuration | [logging/AGENTS.md](logging/AGENTS.md) |
| `utils/` | optional dependency helpers and low-level utilities | [utils/AGENTS.md](utils/AGENTS.md) |

## Current Public API Shape

```python
from ml4t.engineer import (
    compute_features,
    create_dataset_builder,
    feature_catalog,
    MLDatasetBuilder,
    StandardScaler,
    RobustScaler,
)
from ml4t.engineer.config import LabelingConfig, PreprocessingConfig
from ml4t.engineer.labeling import triple_barrier_labels, atr_triple_barrier_labels
from ml4t.engineer.bars import TickBarSampler, VolumeBarSampler, DollarBarSampler
```

## Core Patterns

### Compute features through the registry

```python
from ml4t.engineer import compute_features

result = compute_features(df, ["rsi", "macd", "atr"])
```

### Discover features through metadata

```python
from ml4t.engineer import feature_catalog

feature_catalog.list(category="momentum")
feature_catalog.search("volatility estimator")
feature_catalog.describe("rsi")
```

### Build train/test data without leakage

```python
from ml4t.engineer import create_dataset_builder

builder = create_dataset_builder(features, labels, scaler="robust")
X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.8)
```

## Notes

- `FeatureSelector` has moved out of this library and belongs in `ml4t-diagnostic`
- `store/` exists but is lower-priority than the core feature, labeling, bar, and
  dataset workflows
- AGENTS files are the current navigation surface; older singular filename references
  should be treated as stale
