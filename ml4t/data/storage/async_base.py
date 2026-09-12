"""Async storage backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.data.core.exceptions import StorageError
from ml4t.data.core.models import DataObject


class AsyncStorageBackend(ABC):
    """Abstract base class for async storage backends."""

    @abstractmethod
    async def write(self, data: DataObject) -> str:
        """
        Write data to storage asynchronously.

        Args:
            data: DataObject to write

        Returns:
            Storage key for the written data
        """

    @abstractmethod
    async def read(self, key: str) -> DataObject:
        """
        Read data from storage asynchronously.

        Args:
            key: Storage key

        Returns:
            DataObject from storage
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if data exists in storage asynchronously.

        Args:
            key: Storage key

        Returns:
            True if data exists, False otherwise
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete data from storage asynchronously.

        Args:
            key: Storage key
        """

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys in storage asynchronously.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of storage keys
        """

    async def update(self, key: str, data: DataObject) -> str:
        """
        Update existing data in storage asynchronously.

        Args:
            key: Storage key
            data: New data

        Returns:
            Storage key

        Raises:
            StorageError: If key doesn't exist
        """
        if not await self.exists(key):
            raise StorageError(f"Key {key} does not exist")

        await self.delete(key)
        return await self.write(data)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Default implementation does nothing
        # Subclasses can override for cleanup
        return


# StorageError imported from core.exceptions
