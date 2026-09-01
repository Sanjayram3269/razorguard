"""LLM provider abstraction for the investigation copilot.

Provides a clean interface for LLM interactions with:
- Graceful unavailable behavior when no provider is configured
- Replaceable provider implementations
- Backend-only API key management (never exposed to frontend)
- Timeout handling
- Error isolation

The provider is optional.  When unavailable, the copilot
returns deterministic fallback responses grounded in
RazorGuard evidence.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


# ============================================================
# RESPONSE STRUCTURE
# ============================================================


@dataclass(frozen=True)
class CopilotResponse:
    """Structured copilot response."""

    answer: str
    key_evidence: list[str]
    interpretation: str
    recommended_focus: str
    grounding: str  # "VERIFIED EVIDENCE" or "AI INTERPRETATION" or "INVESTIGATIVE SUGGESTION"


# ============================================================
# PROVIDER INTERFACE
# ============================================================


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider is configured and available."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM.

        Raises:
            ProviderUnavailableError: if the provider is not available.
            ProviderTimeoutError: if the request times out.
            ProviderError: for other provider errors.
        """


# ============================================================
# PROVIDER ERRORS
# ============================================================


class ProviderError(Exception):
    """Base provider error."""


class ProviderUnavailableError(ProviderError):
    """Provider is not configured or available."""


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""


# ============================================================
# OPENAI PROVIDER
# ============================================================


class OpenAIProvider(LLMProvider):
    """OpenAI API provider.

    Uses the openai package if available, otherwise falls back
    to httpx for minimal dependency footprint.
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._timeout = int(os.environ.get("OPENAI_TIMEOUT", "30"))

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        if not self._api_key:
            raise ProviderUnavailableError(
                "OpenAI API key not configured"
            )

        try:
            return self._generate_httpx(
                system_prompt,
                user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ProviderTimeoutError:
            raise
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"OpenAI request failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _generate_httpx(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate using httpx (minimal dependency)."""

        try:
            import httpx
        except ImportError:
            raise ProviderUnavailableError(
                "httpx not installed; cannot call OpenAI API"
            )

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(
                timeout=self._timeout,
            ) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

                return data["choices"][0]["message"]["content"]

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"OpenAI request timed out after {self._timeout}s"
            ) from exc


# ============================================================
# OPENROUTER PROVIDER
# ============================================================


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider.

    Uses OpenRouter's OpenAI-compatible API endpoint.
    Configuration:
        OPENROUTER_API_KEY — required
        OPENROUTER_MODEL  — model to use (default: openrouter/free)
        OPENROUTER_TIMEOUT — request timeout in seconds (default: 30)
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        self._api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._model = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
        self._timeout = int(os.environ.get("OPENROUTER_TIMEOUT", "30"))

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        if not self._api_key:
            raise ProviderUnavailableError(
                "OpenRouter API key not configured"
            )

        try:
            return self._generate_httpx(
                system_prompt,
                user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ProviderTimeoutError:
            raise
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"OpenRouter request failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _generate_httpx(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate using httpx (minimal dependency)."""

        try:
            import httpx
        except ImportError:
            raise ProviderUnavailableError(
                "httpx not installed; cannot call OpenRouter API"
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://razorguard.app",
            "X-Title": "RazorGuard Investigation Copilot",
        }

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(
                timeout=self._timeout,
            ) as client:
                response = client.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

                return data["choices"][0]["message"]["content"]

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"OpenRouter request timed out after {self._timeout}s"
            ) from exc


# ============================================================
# PROVIDER FACTORY
# ============================================================


_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Return the configured LLM provider (singleton).

    Priority:
        1. OpenRouterProvider if OPENROUTER_API_KEY is set
        2. OpenAIProvider if OPENAI_API_KEY is set
        3. NullProvider (unavailable, deterministic fallback)
    """

    global _provider

    if _provider is not None:
        return _provider

    if os.environ.get("OPENROUTER_API_KEY"):
        _provider = OpenRouterProvider()
    elif os.environ.get("OPENAI_API_KEY"):
        _provider = OpenAIProvider()
    else:
        _provider = _NullProvider()

    return _provider


def reset_provider() -> None:
    """Reset the provider singleton (for testing)."""

    global _provider

    _provider = None


class _NullProvider(LLMProvider):
    """Provider that always returns unavailable."""

    def is_available(self) -> bool:
        return False

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        raise ProviderUnavailableError(
            "No LLM provider configured"
        )
