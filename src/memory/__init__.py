"""Memory module — pluggable, provider-agnostic memory for agents.

Quick start
-----------
::

    from src.memory.factory import create_memory_provider

    memory = create_memory_provider()   # defaults to InMemoryProvider (no deps)
    memory.add([{"role": "user", "content": "I prefer low-risk stocks"}],
               user_id="u-123", agent_id="portfolio")
    facts = memory.search("risk preference", user_id="u-123")

Switch to mem0 cloud by setting env vars::

    MEMORY_PROVIDER=mem0
    MEM0_API_KEY=your-api-key

Public surface
--------------
- :class:`~src.memory.base.BaseMemoryProvider` — the provider interface
- :class:`~src.memory.config.MemoryConfig` — resolved configuration
- :func:`~src.memory.config.load_config` — load config from env vars
- :func:`~src.memory.factory.create_memory_provider` — instantiate a provider
"""

from src.memory.base import BaseMemoryProvider
from src.memory.config import MemoryConfig, load_config
from src.memory.factory import create_memory_provider

__all__ = [
    "BaseMemoryProvider",
    "MemoryConfig",
    "load_config",
    "create_memory_provider",
]
