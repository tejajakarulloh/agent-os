"""Per-session write locks to prevent concurrent modifications."""

from __future__ import annotations

import asyncio
from typing import Any


class SessionWriteLock:
    """Async lock per session_key to serialize writes.

    Entries are evicted from the internal dict as soon as the lock is fully
    idle (released with no pending waiters), so the dict does not grow
    unboundedly with the number of unique session keys.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, session_key: str) -> None:
        if session_key not in self._locks:
            self._locks[session_key] = asyncio.Lock()
        await self._locks[session_key].acquire()

    def release(self, session_key: str) -> None:
        if session_key in self._locks:
            lock = self._locks[session_key]
            # Evict if no waiter is queued: the next acquire() will create a
            # fresh lock for this session_key.  If waiters exist, keep the entry
            # so they can acquire the already-released lock.
            if not lock._waiters:
                del self._locks[session_key]
            lock.release()

    async def __aenter__(self) -> SessionWriteLock:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def context(self, session_key: str) -> SessionLockContext:
        """Return a context manager for the session lock."""
        return SessionLockContext(self, session_key)


class SessionLockContext:
    """Context manager for session write lock."""

    def __init__(self, lock_manager: SessionWriteLock, session_key: str) -> None:
        self._lock_manager = lock_manager
        self._session_key = session_key

    async def __aenter__(self) -> None:
        await self._lock_manager.acquire(self._session_key)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._lock_manager.release(self._session_key)
