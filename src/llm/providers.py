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
    Raise ValueError with a clear message when the provider is unknown.
    Unknown model names emit a warning (not an error) since providers accept
    custom / fine-tuned model names not in the known list.
    """
    if provider not in PROVIDERS:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Supported providers: {supported}.\n"
            f"Set via PORTFOLIO_LLM_PROVIDER env var."
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


def infer_provider(model: str) -> str:
    """Infer the provider from a model name using the known-models registry.

    Raises ValueError immediately if the model is not recognised in any
    provider's known list and PORTFOLIO_LLM_PROVIDER is not set as a fallback.
    This gives a clear, actionable error before any LLM call is attempted.
    """
    for provider, models in _KNOWN_MODELS.items():
        if model in models:
            return provider

    # Unknown model — allow explicit provider override via env var
    import os
    explicit = os.getenv("PORTFOLIO_LLM_PROVIDER")
    if explicit:
        if explicit not in PROVIDERS:
            raise ValueError(
                f"PORTFOLIO_LLM_PROVIDER='{explicit}' is not a supported provider. "
                f"Supported: {', '.join(sorted(PROVIDERS))}."
            )
        import warnings
        warnings.warn(
            f"Model '{model}' is not in the known model list. "
            f"Using provider '{explicit}' from PORTFOLIO_LLM_PROVIDER env var. "
            "If the model name is wrong the LLM call will fail.",
            stacklevel=2,
        )
        return explicit

    known_lines = "\n".join(
        f"  {p}: {', '.join(m)}" for p, m in _KNOWN_MODELS.items()
    )
    raise ValueError(
        f"Unknown model '{model}'. Cannot infer provider automatically.\n"
        f"Known models:\n{known_lines}\n\n"
        f"To use a custom model, also set:\n"
        f"  $env:PORTFOLIO_LLM_PROVIDER = \"ollama\"  # or openai / google"
    )


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
