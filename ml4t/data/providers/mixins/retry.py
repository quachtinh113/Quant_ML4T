"""Retry mixin for providers with exponential backoff."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ml4t.data.core.exceptions import NetworkError, RateLimitError

logger = structlog.get_logger()

T = TypeVar("T")


class RetryMixin:
    """Mixin providing retry functionality with exponential backoff.

    Provides automatic retry logic for transient failures like
    network errors and rate limiting.

    Class Variables:
        RETRY_ATTEMPTS: Number of retry attempts
        RETRY_MULTIPLIER: Exponential backoff multiplier
        RETRY_MIN_WAIT: Minimum wait between retries (seconds)
        RETRY_MAX_WAIT: Maximum wait between retries (seconds)
        RETRYABLE_EXCEPTIONS: Exception types that trigger retry

    Example:
        class MyProvider(RetryMixin):
            def fetch_data(self, symbol):
                return self._with_retry(self._do_fetch, symbol)

            def _do_fetch(self, symbol):
                # ... actual fetch logic ...
    """

    RETRY_ATTEMPTS: ClassVar[int] = 3
    RETRY_MULTIPLIER: ClassVar[float] = 1.0
    RETRY_MIN_WAIT: ClassVar[float] = 4.0
    RETRY_MAX_WAIT: ClassVar[float] = 10.0
    RETRYABLE_EXCEPTIONS: ClassVar[tuple[type[Exception], ...]] = (
        NetworkError,
        RateLimitError,
    )

    def _with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Original exception after all retries exhausted
        """

        @retry(
            stop=stop_after_attempt(self.RETRY_ATTEMPTS),
            wait=wait_exponential(
                multiplier=self.RETRY_MULTIPLIER,
                min=self.RETRY_MIN_WAIT,
                max=self.RETRY_MAX_WAIT,
            ),
            retry=retry_if_exception_type(self.RETRYABLE_EXCEPTIONS),
            reraise=True,
        )
        def _retry_wrapper() -> T:
            return func(*args, **kwargs)

        return _retry_wrapper()

    def _create_retry_decorator(
        self,
        attempts: int | None = None,
        min_wait: float | None = None,
        max_wait: float | None = None,
        retryable: tuple[type[Exception], ...] | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Create a custom retry decorator.

        Args:
            attempts: Number of attempts (default: RETRY_ATTEMPTS)
            min_wait: Minimum wait seconds (default: RETRY_MIN_WAIT)
            max_wait: Maximum wait seconds (default: RETRY_MAX_WAIT)
            retryable: Exception types to retry (default: RETRYABLE_EXCEPTIONS)

        Returns:
            Retry decorator configured with specified parameters
        """
        return retry(
            stop=stop_after_attempt(attempts or self.RETRY_ATTEMPTS),
            wait=wait_exponential(
                multiplier=self.RETRY_MULTIPLIER,
                min=min_wait or self.RETRY_MIN_WAIT,
                max=max_wait or self.RETRY_MAX_WAIT,
            ),
            retry=retry_if_exception_type(retryable or self.RETRYABLE_EXCEPTIONS),
            reraise=True,
        )
