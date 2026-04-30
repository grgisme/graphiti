"""FailoverLLMClient — wraps an ordered list of LLMClient providers with
RateLimit/5xx detection, advancement, and per-provider circuit breakers.

Built for donut backlog #1141. The Donut brain ingestion pipeline lost three
large family-roster pastes on 2026-04-30 because Gemini rate-limited mid-
extraction and graphiti dropped them. This wrapper makes that failure mode
recoverable: when one provider rate-limits, the next in the chain handles the
call without any episode loss.

Behavior:
- Tries providers in order on every call (advance only on RateLimit/HTTP 5xx).
- Tracks per-provider failure timestamps; if `circuit_breaker_threshold`
  failures occur within `circuit_breaker_window_sec`, the provider is "opened"
  (skipped) for `circuit_breaker_open_sec` before retrying.
- If ALL providers exhausted in one call, writes a marker file at
  `~/code/donut/state/graphiti-chain-failures.log` for cross-session visibility
  and re-raises the last exception.

The wrapper itself extends LLMClient and only overrides `_generate_response`.
The base class's retry decorator + tracer + token-tracking still apply on the
outer layer; we deliberately do NOT layer additional retries inside the
wrapper since the inner LLMClient.generate_response_with_retry already does
its own back-off and we want failover to advance promptly.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)


# Default cap on failures per provider before opening circuit
_FAILURE_TYPES = (RateLimitError, httpx.HTTPStatusError)


class _CircuitBreaker:
    """Per-provider circuit breaker state."""

    __slots__ = ("name", "failures", "threshold", "window_sec", "open_until_ts")

    def __init__(self, name: str, threshold: int, window_sec: int) -> None:
        self.name = name
        self.failures: deque[float] = deque()
        self.threshold = threshold
        self.window_sec = window_sec
        self.open_until_ts: float = 0.0

    def is_open(self) -> bool:
        return time.monotonic() < self.open_until_ts

    def record_failure(self, open_for_sec: int) -> None:
        now = time.monotonic()
        cutoff = now - self.window_sec
        # Drop expired failures
        while self.failures and self.failures[0] < cutoff:
            self.failures.popleft()
        self.failures.append(now)
        if len(self.failures) >= self.threshold:
            self.open_until_ts = now + open_for_sec
            logger.warning(
                "Circuit OPEN for provider=%s (%d failures in %ds; reopening at %s)",
                self.name,
                len(self.failures),
                self.window_sec,
                datetime.fromtimestamp(time.time() + open_for_sec).isoformat(timespec="seconds"),
            )
            self.failures.clear()

    def record_success(self) -> None:
        # A successful call resets the breaker — half-open semantics.
        self.failures.clear()
        self.open_until_ts = 0.0


class FailoverLLMClient(LLMClient):
    """Drop-in LLMClient that wraps an ordered list of inner clients.

    The wrapped clients are tried in order. Construction order = preference
    order. Use config.fallback_chain to control the order at the call site.
    """

    def __init__(
        self,
        inner: list[tuple[str, LLMClient]],
        *,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_window_sec: int = 300,
        circuit_breaker_open_sec: int = 600,
        notify_on_chain_failure: bool = True,
        notify_path: str | None = None,
    ) -> None:
        if not inner:
            raise ValueError("FailoverLLMClient requires at least one inner provider")
        # Adopt the first inner client's config for self.* attributes
        first_cfg = inner[0][1].config
        super().__init__(first_cfg)
        self.inner = inner
        self.breakers: dict[str, _CircuitBreaker] = {
            name: _CircuitBreaker(name, circuit_breaker_threshold, circuit_breaker_window_sec)
            for name, _ in inner
        }
        self.open_for_sec = circuit_breaker_open_sec
        self.notify_on_chain_failure = notify_on_chain_failure
        self.notify_path = (
            Path(notify_path)
            if notify_path
            else Path(os.path.expanduser("~/code/donut/state/graphiti-chain-failures.log"))
        )

    @property
    def provider_names(self) -> list[str]:
        return [n for n, _ in self.inner]

    def _notify_chain_failure(self, last_exc: BaseException) -> None:
        if not self.notify_on_chain_failure:
            return
        try:
            self.notify_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.notify_path, "a") as f:
                f.write(
                    f"{datetime.now().isoformat(timespec='seconds')}\t"
                    f"chain_exhausted\t{type(last_exc).__name__}\t"
                    f"providers={','.join(self.provider_names)}\t"
                    f"breakers_open={','.join(n for n, b in self.breakers.items() if b.is_open())}\t"
                    f"detail={str(last_exc)[:200]}\n"
                )
        except Exception as e:  # noqa: BLE001
            # Notification is best-effort
            logger.error("FailoverLLMClient: failed to write notify marker: %s", e)

    def set_tracer(self, tracer) -> None:  # type: ignore[no-untyped-def]
        super().set_tracer(tracer)
        # Propagate tracer to inner clients so per-provider span data still works
        for _, inner_client in self.inner:
            try:
                inner_client.set_tracer(tracer)
            except Exception:  # noqa: BLE001
                pass

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,  # type: ignore[assignment]
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Try each provider in order, advancing on RateLimit / 5xx."""
        last_exc: BaseException | None = None
        for name, client in self.inner:
            breaker = self.breakers[name]
            if breaker.is_open():
                logger.info("Skipping provider=%s (circuit open)", name)
                continue
            try:
                # Use the inner's generate_response which carries its own retry +
                # tracer + token tracker. We only intercept terminal failures.
                effective_max = max_tokens if max_tokens is not None else client.max_tokens
                result = await client.generate_response(
                    messages,
                    response_model=response_model,
                    max_tokens=effective_max,
                    model_size=model_size,
                )
                breaker.record_success()
                if name != self.inner[0][0]:
                    logger.info(
                        "FailoverLLMClient: succeeded on fallback provider=%s (primary=%s)",
                        name,
                        self.inner[0][0],
                    )
                return result
            except _FAILURE_TYPES as e:
                last_exc = e
                breaker.record_failure(self.open_for_sec)
                logger.warning(
                    "FailoverLLMClient: %s on provider=%s — advancing chain (%s)",
                    type(e).__name__,
                    name,
                    str(e)[:200],
                )
                continue
            except Exception as e:  # noqa: BLE001
                # Non-rate-limit errors (model misconfig, API key bad, schema
                # error) are not failover-eligible — re-raise as-is.
                logger.error(
                    "FailoverLLMClient: non-failover error on provider=%s — re-raising: %s",
                    name,
                    e,
                )
                raise
        # Chain exhausted
        if last_exc is None:
            last_exc = RuntimeError(
                f"All providers in failover chain are circuit-open: {self.provider_names}"
            )
        self._notify_chain_failure(last_exc)
        logger.critical(
            "FailoverLLMClient: ALL providers exhausted (%s). Last error: %s",
            self.provider_names,
            last_exc,
        )
        raise last_exc

    # The inner clients have their own caches; bypass our layer's cache to
    # avoid duplicated cache hits across the chain.
    def _get_cache_key(self, messages):  # type: ignore[no-untyped-def]
        # Delegate to the active primary
        return self.inner[0][1]._get_cache_key(messages)
