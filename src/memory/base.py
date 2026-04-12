from abc import ABC, abstractmethod
from typing import Optional


class BaseMemoryProvider(ABC):
    """Abstract contract for all memory backends.

    Agents depend only on this interface.  Concrete implementations
    (mem0, Zep, Redis, …) are plugged in at runtime via the factory.

    Terminology
    -----------
    user_id  : stable identifier for the human user (e.g. session id,
               customer id).  Used as the primary memory partition key.
    agent_id : optional identifier for the agent writing/reading memories.
               Allows per-agent memory namespacing within the same user scope.
    memory_id: opaque string returned by the backend when a memory is stored.
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @abstractmethod
    def add(
        self,
        messages: list[dict],
        user_id: str,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> list[str]:
        """Persist a conversation turn and return the assigned memory IDs.

        Parameters
        ----------
        messages:
            List of ``{"role": "user"|"assistant", "content": "..."}`` dicts
            representing the turn to memorise.
        user_id:
            The user partition key.
        agent_id:
            Optional agent namespace within the user partition.
        metadata:
            Arbitrary key/value pairs stored alongside the memory
            (e.g. ``{"source": "ecommerce_support", "order_id": "ORD-123"}``).

        Returns
        -------
        list[str]
            The backend-assigned memory IDs for the stored memories.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search over stored memories.

        Parameters
        ----------
        query:
            Natural-language query used to retrieve relevant memories.
        user_id:
            Restrict search to this user's memories.
        agent_id:
            Further restrict to a specific agent namespace.
        limit:
            Maximum number of memories to return.

        Returns
        -------
        list[dict]
            Each dict contains at minimum ``{"id": str, "memory": str,
            "score": float}``.  Additional backend fields are passed
            through unchanged.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        """Return all stored memories for a user (optionally filtered by agent).

        Returns
        -------
        list[dict]
            Same shape as :meth:`search` results but without a score field.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete a single memory by its ID."""
        raise NotImplementedError

    @abstractmethod
    def delete_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """Delete all memories for a user (optionally scoped to an agent)."""
        raise NotImplementedError
