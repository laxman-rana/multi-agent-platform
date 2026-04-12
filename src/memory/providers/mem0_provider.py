"""mem0 cloud memory provider.

Requires: pip install mem0ai
Activate: MEMORY_PROVIDER=mem0 and MEM0_API_KEY=<key>
"""

from __future__ import annotations

from typing import Optional

from src.memory.base import BaseMemoryProvider
from src.memory.config import MemoryConfig


def _user_filter(user_id: str) -> dict:
    """mem0 AND-filter scoped to a single user_id."""
    return {"AND": [{"user_id": user_id}]}


class Mem0Provider(BaseMemoryProvider):
    """mem0 cloud-backed memory provider."""

    def __init__(self, config: MemoryConfig) -> None:
        try:
            from mem0 import MemoryClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "mem0ai is not installed. Run: pip install mem0ai"
            ) from exc

        if not config.mem0_api_key:
            raise ValueError(
                "MEM0_API_KEY is required. Set the MEM0_API_KEY environment variable."
            )

        self._memory = MemoryClient(api_key=config.mem0_api_key)

    def add(
        self,
        messages: list[dict],
        user_id: str,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> list[str]:
        kwargs: dict = {"user_id": user_id}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if metadata:
            kwargs["metadata"] = metadata
        result = self._memory.add(messages, **kwargs)
        results = result.get("results", result) if isinstance(result, dict) else result
        return [r["id"] for r in results if "id" in r]

    def search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        # agent_id is not indexed as a filterable field in mem0 cloud.
        result = self._memory.search(query, filters=_user_filter(user_id), top_k=limit)
        return result.get("results", result) if isinstance(result, dict) else result

    def get_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        # agent_id is not indexed as a filterable field in mem0 cloud.
        result = self._memory.get_all(filters=_user_filter(user_id))
        return result.get("results", result) if isinstance(result, dict) else result

    def delete(self, memory_id: str) -> None:
        self._memory.delete(memory_id)

    def delete_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> None:
        self._memory.delete_all(filters=_user_filter(user_id))
