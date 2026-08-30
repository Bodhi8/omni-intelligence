from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .types import Generation


class ProviderError(RuntimeError):
    """Raised when a model provider cannot return a usable generation."""


class ModelProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> Generation:
        raise NotImplementedError


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {exc.code} from provider: {body[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"Provider request failed: {exc}") from exc


def _estimated_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    if input_price_per_million is None or output_price_per_million is None:
        return None
    return (
        input_tokens * input_price_per_million / 1_000_000
        + output_tokens * output_price_per_million / 1_000_000
    )


@dataclass
class ReplayProvider(ModelProvider):
    name: str
    model: str
    response: str
    latency_seconds: float = 0.0

    async def generate(self, prompt: str, system: str = "") -> Generation:
        return Generation(
            provider=self.name,
            model=self.model,
            text=self.response,
            latency_seconds=self.latency_seconds,
        )


@dataclass
class OpenAICompatibleProvider(ModelProvider):
    name: str
    model: str
    api_key_env: str
    base_url: str
    timeout: int = 120
    temperature: float = 0.2
    max_tokens: int = 1600
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    async def generate(self, prompt: str, system: str = "") -> Generation:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"Missing environment variable: {self.api_key_env}")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        started = time.perf_counter()
        data = await asyncio.to_thread(
            _post_json,
            f"{self.base_url.rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {key}"},
            payload,
            self.timeout,
        )
        latency = time.perf_counter() - started
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI-compatible response did not contain message content") from exc
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        return Generation(
            provider=self.name,
            model=self.model,
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=_estimated_cost(
                input_tokens,
                output_tokens,
                self.input_price_per_million,
                self.output_price_per_million,
            ),
            raw_usage=usage,
        )


@dataclass
class AnthropicProvider(ModelProvider):
    name: str
    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"
    timeout: int = 120
    temperature: float = 0.2
    max_tokens: int = 1600
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    async def generate(self, prompt: str, system: str = "") -> Generation:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"Missing environment variable: {self.api_key_env}")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system:
            payload["system"] = system
        started = time.perf_counter()
        data = await asyncio.to_thread(
            _post_json,
            f"{self.base_url.rstrip('/')}/messages",
            {"x-api-key": key, "anthropic-version": self.api_version},
            payload,
            self.timeout,
        )
        latency = time.perf_counter() - started
        blocks = data.get("content", [])
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        if not text:
            raise ProviderError("Anthropic response did not contain text content")
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        return Generation(
            provider=self.name,
            model=self.model,
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=_estimated_cost(
                input_tokens,
                output_tokens,
                self.input_price_per_million,
                self.output_price_per_million,
            ),
            raw_usage=usage,
        )


@dataclass
class GeminiProvider(ModelProvider):
    name: str
    model: str
    api_key_env: str = "GEMINI_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout: int = 120
    temperature: float = 0.2
    max_tokens: int = 1600
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    async def generate(self, prompt: str, system: str = "") -> Generation:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"Missing environment variable: {self.api_key_env}")
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        encoded_model = urllib.parse.quote(self.model, safe="")
        encoded_key = urllib.parse.quote(key, safe="")
        url = f"{self.base_url.rstrip('/')}/models/{encoded_model}:generateContent?key={encoded_key}"
        started = time.perf_counter()
        data = await asyncio.to_thread(_post_json, url, {}, payload, self.timeout)
        latency = time.perf_counter() - started
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response did not contain text content") from exc
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")
        return Generation(
            provider=self.name,
            model=self.model,
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=_estimated_cost(
                input_tokens,
                output_tokens,
                self.input_price_per_million,
                self.output_price_per_million,
            ),
            raw_usage=usage,
        )


def provider_from_mapping(spec: dict[str, Any]) -> ModelProvider:
    kind = spec.get("type", "openai_compatible")
    allowed = {
        key: value
        for key, value in spec.items()
        if key not in {"type", "role", "enabled"}
    }
    if kind == "openai_compatible":
        return OpenAICompatibleProvider(**allowed)
    if kind == "anthropic":
        return AnthropicProvider(**allowed)
    if kind == "gemini":
        return GeminiProvider(**allowed)
    if kind == "replay":
        return ReplayProvider(**allowed)
    raise ValueError(f"Unsupported provider type: {kind}")
