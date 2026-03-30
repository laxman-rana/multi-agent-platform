import os
from abc import ABC, abstractmethod


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear configuration error."""

    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


# ---------------------------------------------------------------------------
# Default models — override any of these at runtime via:
#   env var   PORTFOLIO_LLM_MODEL=<name>
#   CLI flag  --model <name>
# ---------------------------------------------------------------------------
_DEFAULT_MODELS = {
    "ollama": "gpt-oss:120b",
    "openai": "gpt-4o",
    "google": "gemini-1.5-pro",
}

# Known / tested model names per provider.
# This is not exhaustive — providers may accept other model names — but unknown
# names trigger a warning so users can catch typos before the LLM call fails.
_KNOWN_MODELS: dict[str, list[str]] = {
    "ollama": ["gpt-oss:120b", "llama3", "llama3:70b", "mistral", "mixtral", "gemma2", "phi3"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-2.0-flash"],
}


def validate_provider_model(provider: str, model: str | None = None) -> None:
    """
    Raise ValueError with a clear message when the provider or model is unknown.

    Provider must be in PROVIDERS.  Model is validated as a known name if
    provided; unknown models emit a warning (not an error) since providers
    accept custom / fine-tuned model names that are not in the list.
    """
    if provider not in PROVIDERS:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Supported providers: {supported}.\n"
            f"Set via --provider flag or PORTFOLIO_LLM_PROVIDER env var."
        )
    if model is not None:
        known = _KNOWN_MODELS.get(provider, [])
        if known and model not in known:
            import warnings
            warnings.warn(
                f"Model '{model}' is not in the known model list for provider '{provider}'. "
                f"Known models: {', '.join(known)}. "
                "Continuing anyway — if the model name is wrong the LLM call will fail.",
                stacklevel=3,
            )


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_llm(self, tools=None, callbacks=None, model: str | None = None):
        """Return the configured LLM, optionally bound to tools and callbacks.

        Args:
            model: explicit model name override; when None the provider reads
                   from the PORTFOLIO_LLM_MODEL env var (or its built-in default).
        """
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Provider for Ollama-hosted models."""

    def get_llm(self, tools=None, callbacks=None, model: str | None = None):
        from langchain_ollama import ChatOllama

        model = model or os.getenv("PORTFOLIO_LLM_MODEL", _DEFAULT_MODELS["ollama"])
        llm = ChatOllama(
            model=model,
            temperature=0,
            base_url="https://ollama.com",
            client_kwargs={"headers": {"Authorization": f"Bearer {require_env('OLLAMA_API_KEY')}"}},
            reasoning=False,
            callbacks=callbacks,
        )
        if tools:
            llm = llm.bind_tools(tools)
        return llm


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models."""

    def get_llm(self, tools=None, callbacks=None, model: str | None = None):
        from langchain_openai import ChatOpenAI

        model = model or os.getenv("PORTFOLIO_LLM_MODEL", _DEFAULT_MODELS["openai"])
        llm = ChatOpenAI(model=model, temperature=0.7, callbacks=callbacks)
        if tools:
            llm = llm.bind_tools(tools)
        return llm


class GoogleProvider(LLMProvider):
    """Provider for Google Gemini models."""

    def get_llm(self, tools=None, callbacks=None, model: str | None = None):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model or os.getenv("PORTFOLIO_LLM_MODEL", _DEFAULT_MODELS["google"])
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.7, callbacks=callbacks)
        if tools:
            llm = llm.bind_tools(tools)
        return llm


PROVIDERS = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}


def get_provider(model_name="ollama"):
    """Return a provider instance for the requested model backend."""

    provider_class = PROVIDERS.get(model_name)
    if not provider_class:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"Unsupported provider '{model_name}'. "
            f"Supported providers: {supported}."
        )
    return provider_class()


def get_llm(model_name="ollama", tools=None, callbacks=None, model: str | None = None):
    """Return a configured LLM from the selected provider.

    Args:
        model_name: provider key ("ollama", "openai", "google").
        model:      explicit LLM model name; overrides PORTFOLIO_LLM_MODEL env var.
    """
    provider = get_provider(model_name)
    return provider.get_llm(tools=tools, callbacks=callbacks, model=model)
