"""Retry utility with exponential backoff and jitter.

Provides a shared retry_with_backoff async function used by all external
API calls (GitHub, Bedrock, authoritative source fetches) throughout the
watchdog system.
"""

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,),
) -> T:
    """Execute an async operation with exponential backoff retry.

    Attempts the operation once, then retries up to ``max_retries`` times
    on failure, for a maximum of ``max_retries + 1`` total attempts (4 by
    default). Each retry waits with exponential backoff plus jitter.

    Delay formula: ``min(base_delay * 2^attempt + random(0, base_delay), max_delay)``

    Args:
        operation: An async callable (no arguments) that returns a value of type T.
        max_retries: Maximum number of retry attempts after the initial try.
        base_delay: Base delay in seconds for backoff calculation.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Tuple of exception types that trigger a retry.

    Returns:
        The successful result from ``operation``.

    Raises:
        The last exception encountered if all retries are exhausted.
    """
    last_exception: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt < max_retries:
                delay = min(
                    base_delay * (2 ** attempt) + random.uniform(0, base_delay),
                    max_delay,
                )
                await asyncio.sleep(delay)
    raise last_exception  # type: ignore[misc]
