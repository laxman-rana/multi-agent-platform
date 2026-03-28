from abc import ABC, abstractmethod

class TelemetryLogger(ABC):
    """Abstract base class for telemetry logging."""

    @abstractmethod
    def log_event(self, event_name: str, data: dict):
        """Log a generic event."""
        pass

    @abstractmethod
    def log_tool_usage(self, tool_name: str, input_data: dict, output_data: dict):
        """Log tool usage details."""
        pass

    @abstractmethod
    def log_llm_interaction(self, prompt: str, response: str):
        """Log interactions with the LLM."""
        pass
