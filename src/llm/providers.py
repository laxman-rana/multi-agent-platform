import os
from abc import ABC, abstractmethod


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear configuration error."""

    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_llm(self, tools=None, callbacks=None):
        """Return the configured LLM, optionally bound to tools and callbacks."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Provider for Ollama-hosted models."""

    def get_llm(self, tools=None, callbacks=None):
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="gpt-oss:120b",
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

    def get_llm(self, tools=None, callbacks=None):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o", temperature=0.7, callbacks=callbacks)
        if tools:
            llm = llm.bind_tools(tools)
        return llm


class GoogleProvider(LLMProvider):
    """Provider for Google Gemini models."""

    def get_llm(self, tools=None, callbacks=None):
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, callbacks=callbacks)
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
        raise ValueError(f"Unknown model provider: {model_name}")
    return provider_class()


def get_llm(model_name="ollama", tools=None, callbacks=None):
    """Return a configured LLM from the selected provider."""

    provider = get_provider(model_name)
    return provider.get_llm(tools=tools, callbacks=callbacks)
