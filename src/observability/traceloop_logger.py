import os
from .base import TelemetryLogger
from traceloop.sdk import Traceloop

class TraceLoopLogger(TelemetryLogger):
    """Concrete implementation of TelemetryLogger using TraceLoop SDK."""

    def __init__(self):
        """Initialize TraceLoopLogger from environment without breaking startup."""

        self.enabled = False
        self.init_error = None

        api_key = os.getenv("TRACELOOP_API_KEY")
        if not api_key:
            self.init_error = "TRACELOOP_API_KEY is not set"
            return

        try:
            Traceloop.init(disable_batch=True)
            self.enabled = True
        except Exception as exc:
            self.init_error = str(exc)

    def log_event(self, event_name: str, data: dict):
        if not self.enabled:
            return
        print(f"[TraceLoop] Logging event: {event_name} with data: {data}")

    def log_tool_usage(self, tool_name: str, input_data: dict, output_data: dict):
        if not self.enabled:
            return
        print(f"[TraceLoop] Tool used: {tool_name}, Input: {input_data}, Output: {output_data}")

    def log_llm_interaction(self, prompt: str, response: str):
        if not self.enabled:
            return
        print(f"[TraceLoop] LLM Interaction - Prompt: {prompt}, Response: {response}")
