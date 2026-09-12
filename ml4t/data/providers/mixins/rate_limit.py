"""Rate limiting mixin for providers."""

from __future__ import annotations

from typing import Any, ClassVar

import structlog

from ml4t.data.utils.global_rate_limit import global_rate_limit_manager

logger = structlog.get_logger()


class RateLimitMixin:
    """Mixin providing rate limiting functionality.

    Providers using this mixin automatically get rate limiting
    through the global rate limit manager.

    Class Variables:
        DEFAULT_RATE_LIMIT: Default (calls, period_seconds) tuple

    Example:
        class MyProvider(RateLimitMixin):
            DEFAULT_RATE_LIMIT = (100, 60.0)  # 100 calls per minute

            def fetch_data(self, symbol):
                self._acquire_rate_limit()
                # ... fetch data ...
    """

    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (60, 60.0)

    # Instance attributes (set by __init__ or init_rate_limit)
    rate_limiter: Any
    _rate_limit_initialized: bool = False

    def init_rate_limit(
        self,
        provider_name: str,
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        """Initialize rate limiting.

        Args:
            provider_name: Name of the provider (for rate limit tracking)
            rate_limit: Optional (calls, period) override
        """
        rate_calls, rate_period = rate_limit or self.DEFAULT_RATE_LIMIT
        self.rate_limiter = global_rate_limit_manager.get_rate_limiter(
            provider_name=provider_name,
            max_calls=rate_calls,
            period=rate_period,
        )
        self._rate_limit_initialized = True

        logger.debug(
            "Rate limiter initialized",
            provider=provider_name,
            max_calls=rate_calls,
            period=rate_period,
        )

    def _acquire_rate_limit(self, blocking: bool = True) -> bool:
        """Acquire rate limiting permission.

        Args:
            blocking: If True, block until permission granted

        Returns:
            True if permission granted, False if non-blocking and denied
        """
        if not self._rate_limit_initialized:
            return True  # No rate limiting if not initialized

        return self.rate_limiter.acquire(blocking=blocking)

    def _get_rate_limit_status(self) -> dict[str, Any]:
        """Get current rate limit status.

        Returns:
            Dict with rate limit information
        """
        if not self._rate_limit_initialized:
            return {"initialized": False}

        return {
            "initialized": True,
            "remaining_calls": getattr(self.rate_limiter, "remaining", None),
            "reset_time": getattr(self.rate_limiter, "reset_time", None),
        }
