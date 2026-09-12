"""Provider mixins for composable behavior.

Mixins provide reusable functionality that can be added to providers:
- RateLimitMixin: API rate limiting
- RetryMixin: Automatic retry with backoff
- CircuitBreakerMixin: Circuit breaker pattern
- ValidationMixin: OHLCV data validation
- SessionMixin: HTTP session management (sync)
- AsyncSessionMixin: HTTP session management (async)

Usage:
    # Sync provider
    class MyProvider(RateLimitMixin, RetryMixin, ValidationMixin):
        @property
        def name(self) -> str:
            return "my_provider"

        def fetch_ohlcv(self, symbol, start, end, frequency="daily"):
            self._acquire_rate_limit()  # From RateLimitMixin
            data = self._fetch_with_retry(...)  # From RetryMixin
            return self._validate_ohlcv(data)  # From ValidationMixin

    # Async provider
    class MyAsyncProvider(RateLimitMixin, AsyncSessionMixin, ValidationMixin):
        async def fetch_ohlcv_async(self, symbol, start, end, frequency="daily"):
            self._acquire_rate_limit()
            data = await self._aget_json(url)
            return self._validate_ohlcv(data)
"""

from ml4t.data.providers.mixins.async_session import AsyncSessionMixin
from ml4t.data.providers.mixins.circuit_breaker import CircuitBreakerMixin
from ml4t.data.providers.mixins.rate_limit import RateLimitMixin
from ml4t.data.providers.mixins.retry import RetryMixin
from ml4t.data.providers.mixins.session import SessionMixin
from ml4t.data.providers.mixins.validation import ValidationMixin

__all__ = [
    "RateLimitMixin",
    "RetryMixin",
    "CircuitBreakerMixin",
    "ValidationMixin",
    "SessionMixin",
    "AsyncSessionMixin",
]
