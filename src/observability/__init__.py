from functools import lru_cache

from src.observability.traceloop_logger import TraceLoopLogger


@lru_cache(maxsize=1)
def get_telemetry_logger() -> TraceLoopLogger:
    """Return the shared TraceLoopLogger singleton. Safe to call from any agent."""
    return TraceLoopLogger()
