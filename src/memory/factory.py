"""Memory provider factory.

Usage (anywhere in the codebase)::

    from src.memory.factory import create_memory_provider

    memory = create_memory_provider()          # reads MEMORY_PROVIDER env var
    memory.add(messages, user_id="u-123")
    results = memory.search("refund policy", user_id="u-123")

To use an explicit config (e.g. in tests)::

    from src.memory.config import MemoryConfig
    from src.memory.factory import create_memory_provider

    cfg = MemoryConfig(provider="in_memory")
    memory = create_memory_provider(config=cfg)

Extending with a new provider
------------------------------
1. Create ``src/memory/providers/my_provider.py`` implementing
   :class:`~src.memory.base.BaseMemoryProvider`.
2. Add the key to ``_REGISTRY`` below.
3. Set ``MEMORY_PROVIDER=my_provider`` in the environment.
No other files need to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.memory.base import BaseMemoryProvider
from src.memory.config import MemoryConfig, load_config

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Maps the MEMORY_PROVIDER env-var value to a callable that accepts a
# MemoryConfig and returns a BaseMemoryProvider instance.
# Imports are deferred so that optional dependencies (mem0, qdrant-client, …)
# are only imported when that specific provider is actually selected.

def _make_in_memory(config: MemoryConfig) -> BaseMemoryProvider:
    from src.memory.providers.in_memory import InMemoryProvider
    return InMemoryProvider()


def _make_mem0(config: MemoryConfig) -> BaseMemoryProvider:
    from src.memory.providers.mem0_provider import Mem0Provider
    return Mem0Provider(config)


_REGISTRY: dict[str, object] = {
    "in_memory": _make_in_memory,
    "mem0":      _make_mem0,
}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_memory_provider(config: MemoryConfig | None = None) -> BaseMemoryProvider:
    """Instantiate and return the configured memory provider.

    Parameters
    ----------
    config:
        Explicit :class:`~src.memory.config.MemoryConfig` to use.
        When ``None``, configuration is read from environment variables via
        :func:`~src.memory.config.load_config`.

    Returns
    -------
    BaseMemoryProvider
        A ready-to-use memory provider instance.

    Raises
    ------
    ValueError
        If the requested provider is not registered.
    """
    if config is None:
        config = load_config()

    factory_fn = _REGISTRY.get(config.provider)
    if factory_fn is None:
        registered = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"No factory registered for memory provider '{config.provider}'. "
            f"Registered providers: {registered}."
        )

    return factory_fn(config)  # type: ignore[operator]
