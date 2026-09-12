"""Crypto data management for ML4T book.

This module provides a unified interface for downloading and managing
cryptocurrency data from Binance Public Data.

Example:
    >>> from ml4t.data.crypto import CryptoDataManager
    >>> manager = CryptoDataManager.from_config("configs/ml4t_etfs.yaml")
    >>> manager.download_premium_index()
    >>> premium = manager.load_premium_index()
"""

from ml4t.data.crypto.downloader import CryptoDataManager

__all__ = ["CryptoDataManager"]
