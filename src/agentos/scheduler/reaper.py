"""Session reaper — cleans up expired isolated cron sessions."""

from __future__ import annotations

import logging
import time

from agentos.session.runtime_state import evict_session_runtime_state

logger = logging.getLogger(__name__)


def _is_isolated_cron_session(key: str) -> bool:
    """Return True if key matches the pattern cron:<job_id>:run:<run_id>."""
    parts = key.split(":")
    return len(parts) == 4 and parts[0] == "cron" and parts[2] == "run"


class SessionReaper:
    """Periodically deletes expired isolated cron sessions from the session store."""

    DEFAULT_RETENTION_SECONDS = 86400  # 24 hours
    MIN_REAP_INTERVAL = 300  # 5 minutes
    PAGE_SIZE = 500
    MAX_PAGES = 200  # backstop, so a pathological store cannot spin forever

    def __init__(self, session_store, retention_seconds: int = DEFAULT_RETENTION_SECONDS) -> None:
        self._session_store = session_store
        self._retention = retention_seconds
        self._last_reap: float = 0  # monotonic time

    async def maybe_reap(self) -> None:
        """Reap if MIN_REAP_INTERVAL has elapsed since the last reap."""
        now = time.monotonic()
        if now - self._last_reap < self.MIN_REAP_INTERVAL:
            return
        self._last_reap = now
        await self._do_reap()

    async def _do_reap(self) -> None:
        """Delete expired isolated cron sessions."""
        if self._session_store is None:
            return

        cutoff_ms = int((time.time() - self._retention) * 1000)

        to_delete: list[str] = []
        # list_sessions defaults to the 100 most recently updated sessions, which
        # is precisely the set that is *not* expired — so the whole store has to
        # be walked. Collect first, delete after: deleting mid-sweep would shift
        # the offsets of the pages still to come.
        for page in range(self.MAX_PAGES):
            sessions = await self._session_store.list_sessions(
                limit=self.PAGE_SIZE,
                offset=page * self.PAGE_SIZE,
            )
            for session in sessions:
                key = getattr(session, "session_key", None) or getattr(session, "key", None)
                updated_at = getattr(session, "updated_at", None)
                if key and updated_at is not None:
                    if _is_isolated_cron_session(key) and updated_at < cutoff_ms:
                        to_delete.append(key)
            if len(sessions) < self.PAGE_SIZE:
                break
        else:
            logger.warning("reaper.page_cap_reached pages=%d", self.MAX_PAGES)

        for key in to_delete:
            # Same leak as the other removal paths: the reaper talks straight
            # to storage, so the process-global runtime state keyed by session
            # has to be dropped here too.
            evict_session_runtime_state(key)
            await self._session_store.delete_session(key)

        if to_delete:
            logger.info("reaper.deleted count=%d", len(to_delete))
