"""Memory module configuration.

Environment variables
---------------------
MEMORY_PROVIDER
    Which memory library to use.
    Values : ``mem0`` | ``in_memory``
    Default: ``in_memory``

MEM0_API_KEY
    API key for the mem0 cloud service.  Required when MEMORY_PROVIDER=mem0.
"""

import os
from dataclasses import dataclass


SUPPORTED_PROVIDERS = ("mem0", "in_memory")


@dataclass
class MemoryConfig:
    """Resolved configuration for the memory module.

    Construct via :func:`load_config` to read from environment variables,
    or instantiate directly in tests to inject explicit values.
    """

    provider: str = "in_memory"
    """Which memory library to use: ``mem0`` or ``in_memory``."""

    mem0_api_key: str | None = None
    """mem0 cloud API key.  Required when provider is ``mem0``."""


def load_config() -> MemoryConfig:
    """Build a :class:`MemoryConfig` from environment variables."""
    provider = os.getenv("MEMORY_PROVIDER", "in_memory").lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported MEMORY_PROVIDER '{provider}'. "
            f"Supported values: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    return MemoryConfig(
        provider=provider,
        mem0_api_key=os.getenv("MEM0_API_KEY") or None,
    )
