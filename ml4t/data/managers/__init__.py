"""Manager modules for DataManager decomposition.

This package contains focused manager classes that handle specific responsibilities,
extracted from the original monolithic DataManager class.

Architecture:
    ConfigManager → ProviderManager → FetchManager
                                          ↓
                        BatchManager ← StorageManager → MetadataManager → BulkManager

Usage:
    The managers can be used independently or through the DataManager facade:

    >>> from ml4t.data.managers import ConfigManager, ProviderManager, FetchManager
    >>> config_mgr = ConfigManager()
    >>> provider_mgr = ProviderManager(config_mgr.config)
    >>> router = ProviderRouter()
    >>> router.setup_default_patterns()
    >>> fetch_mgr = FetchManager(provider_mgr, router)
    >>> df = fetch_mgr.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")

    Or through the DataManager facade (preferred):

    >>> from ml4t.data import DataManager
    >>> dm = DataManager()
    >>> df = dm.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")
"""

from ml4t.data.managers.batch_manager import BatchManager
from ml4t.data.managers.bulk_manager import BulkManager
from ml4t.data.managers.config_manager import ConfigManager
from ml4t.data.managers.fetch_manager import FetchManager
from ml4t.data.managers.metadata_manager import MetadataManager
from ml4t.data.managers.provider_manager import ProviderManager, ProviderRouter
from ml4t.data.managers.storage_manager import StorageManager

__all__ = [
    "BatchManager",
    "BulkManager",
    "ConfigManager",
    "FetchManager",
    "MetadataManager",
    "ProviderManager",
    "ProviderRouter",
    "StorageManager",
]
