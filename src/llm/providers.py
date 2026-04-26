import os
import warnings
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
    "ollama": "gpt-oss:20b",
    "openai": "gpt-4o",
    "google": "gemini-1.5-pro",
}

# Lightweight model hints per provider.
# These are examples and validation hints, not a source of truth for all models
# available from each provider.
_MODEL_HINTS: dict[str, list[str]] = {
    "ollama": ["gpt-oss:20b", "gpt-oss:120b", "deepseek-v4-flash", "glm-5.1", "qwen3-next:80b"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
    "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
}

_PROVIDER_PREFIX_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("openai", ("gpt-", "o1", "o3", "o4")),
    ("google", ("gemini-",)),
]


def default_model_for(provider: str) -> str:
    try:
        return _DEFAULT_MODELS[provider]
    except KeyError as exc:
        supported = ", ".join(sorted(_DEFAULT_MODELS))
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers: {supported}."
        ) from exc


def validate_provider_model(provider: str, model: str | None = None) -> None:
    """
    Raise ValueError with a clear message when the provider is unknown.
    Unknown model names emit a warning (not an error) since providers accept
    custom model names and provider catalogs change frequently.
    """
    if provider not in PROVIDERS:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Supported providers: {supported}.\n"
            f"Set via PORTFOLIO_LLM_PROVIDER env var."
        )
    if model is not None:
        hints = _MODEL_HINTS.get(provider, [])
        if hints and model not in hints:
            warnings.warn(
                f"Model '{model}' is not in the hint list for provider '{provider}'. "
                f"Examples for this provider: {', '.join(hints)}. "
                "Continuing anyway — if the model name is wrong the LLM call will fail.",
                stacklevel=3,
            )


def infer_provider(model: str) -> str:
    """Infer the provider from a model name using lightweight heuristics.

    Resolution order:
    1. Explicit PORTFOLIO_LLM_PROVIDER env var, if set
    2. Prefix-based heuristics for well-known provider namespaces
    3. Fallback to 'ollama' for everything else

    This keeps the app flexible when providers add new model names.
    """
    explicit = os.getenv("PORTFOLIO_LLM_PROVIDER")
    if explicit:
        if explicit not in PROVIDERS:
            raise ValueError(
                f"PORTFOLIO_LLM_PROVIDER='{explicit}' is not a supported provider. "
                f"Supported: {', '.join(sorted(PROVIDERS))}."
            )
        return explicit

    lower_model = model.lower().strip()
    for provider, prefixes in _PROVIDER_PREFIX_HINTS:
        if lower_model.startswith(prefixes):
            return provider

    return "ollama"


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    max_concurrency: int = 10
    """Maximum number of concurrent LLM calls this provider can handle.
    Cloud providers (OpenAI, Google) support many parallel requests.
    Override to a lower value for providers with strict concurrency limits.
    """

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

    max_concurrency: int = 1  # local Ollama processes one request at a time

    def get_llm(self, tools=None, callbacks=None, model: str | None = None):
        from langchain_ollama import ChatOllama

        model = model or os.getenv("PORTFOLIO_LLM_MODEL", default_model_for("ollama"))
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

        model = model or os.getenv("PORTFOLIO_LLM_MODEL", default_model_for("openai"))
        llm = ChatOpenAI(model=model, temperature=0.7, callbacks=callbacks)
        if tools:
            llm = llm.bind_tools(tools)
        return llm


class GoogleProvider(LLMProvider):
    """Provider for Google Gemini models."""

    def get_llm(self, tools=None, callbacks=None, model: str | None = None):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model or os.getenv("PORTFOLIO_LLM_MODEL", default_model_for("google"))
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
