"""In-memory (dict-backed) memory provider.

A zero-dependency stub used for local development and unit testing.
No external services or installs required.

Memories are stored in a plain Python dict keyed by ``(user_id, agent_id)``.
All data is lost when the process exits — use a persistent provider (mem0,
Zep, …) for production.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Optional

from src.memory.base import BaseMemoryProvider


class InMemoryProvider(BaseMemoryProvider):
    """Volatile, dict-backed memory provider for dev / testing."""

    def __init__(self) -> None:
        # { (user_id, agent_id) -> { memory_id -> {"id", "memory", "metadata"} } }
        self._store: dict[tuple[str, str | None], dict[str, dict]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        messages: list[dict],
        user_id: str,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> list[str]:
        """Persist each message as a separate memory entry."""
        key = (user_id, agent_id)
        ids: list[str] = []
        for msg in messages:
            memory_id = str(uuid.uuid4())
            self._store[key][memory_id] = {
                "id":       memory_id,
                "memory":   msg.get("content", ""),
                "role":     msg.get("role", ""),
                "metadata": metadata or {},
            }
            ids.append(memory_id)
        return ids

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Naive substring search — sufficient for dev/test; not semantic."""
        key = (user_id, agent_id)
        memories = list(self._store[key].values())
        query_lower = query.lower()
        matched = [
            {**m, "score": 1.0}
            for m in memories
            if query_lower in m["memory"].lower()
        ]
        # Fall back to returning the most-recent memories when nothing matches.
        if not matched:
            matched = [{**m, "score": 0.0} for m in memories]
        return matched[:limit]

    def get_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        key = (user_id, agent_id)
        return list(self._store[key].values())

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, memory_id: str) -> None:
        for partition in self._store.values():
            if memory_id in partition:
                del partition[memory_id]
                return

    def delete_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> None:
        key = (user_id, agent_id)
        self._store[key].clear()
