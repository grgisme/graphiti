"""Tests for FailoverLLMClient (donut backlog #1141)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError

from src.services.failover_llm_client import FailoverLLMClient, _CircuitBreaker


def _mock_client(name: str) -> MagicMock:
    """Build a mock LLMClient suitable for FailoverLLMClient inner list."""
    cfg = CoreLLMConfig(api_key='fake', model='fake-model', small_model='fake-small')
    m = MagicMock(spec=LLMClient)
    m.config = cfg
    m.model = 'fake-model'
    m.small_model = 'fake-small'
    m.temperature = 0.0
    m.max_tokens = 4096
    m._get_cache_key = MagicMock(return_value=f'key-{name}')
    m.set_tracer = MagicMock()
    m.generate_response = AsyncMock(return_value={'ok': True, 'name': name})
    return m


@pytest.mark.asyncio
async def test_primary_succeeds():
    """When primary returns successfully, the fallback never fires."""
    a, b = _mock_client('a'), _mock_client('b')
    fl = FailoverLLMClient([('a', a), ('b', b)])
    result = await fl._generate_response([], None, None, ModelSize.medium)
    assert result['name'] == 'a'
    a.generate_response.assert_awaited_once()
    b.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_advances_on_rate_limit():
    """RateLimitError on primary → call routes to next provider."""
    a, b, c = _mock_client('a'), _mock_client('b'), _mock_client('c')
    a.generate_response = AsyncMock(side_effect=RateLimitError('a rate-limited'))
    fl = FailoverLLMClient([('a', a), ('b', b), ('c', c)])
    result = await fl._generate_response([], None, None, ModelSize.medium)
    assert result['name'] == 'b'
    a.generate_response.assert_awaited_once()
    b.generate_response.assert_awaited_once()
    c.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_advances_on_5xx():
    """HTTP 5xx is also failover-eligible."""
    a, b = _mock_client('a'), _mock_client('b')
    fake_resp = MagicMock(status_code=503)
    a.generate_response = AsyncMock(
        side_effect=httpx.HTTPStatusError('upstream down', request=MagicMock(), response=fake_resp)
    )
    fl = FailoverLLMClient([('a', a), ('b', b)])
    result = await fl._generate_response([], None, None, ModelSize.medium)
    assert result['name'] == 'b'


@pytest.mark.asyncio
async def test_non_failover_error_re_raised():
    """A non-rate-limit, non-5xx error (bad config etc.) propagates immediately."""
    a, b = _mock_client('a'), _mock_client('b')
    a.generate_response = AsyncMock(side_effect=ValueError('bad config'))
    fl = FailoverLLMClient([('a', a), ('b', b)])
    with pytest.raises(ValueError, match='bad config'):
        await fl._generate_response([], None, None, ModelSize.medium)
    b.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_chain_exhausted_raises_last(tmp_path):
    """When ALL providers rate-limit, last exception is re-raised + marker file written."""
    a, b = _mock_client('a'), _mock_client('b')
    a.generate_response = AsyncMock(side_effect=RateLimitError('a'))
    b.generate_response = AsyncMock(side_effect=RateLimitError('b'))
    notify = tmp_path / 'chain-failures.log'
    fl = FailoverLLMClient(
        [('a', a), ('b', b)],
        notify_on_chain_failure=True,
        notify_path=str(notify),
    )
    with pytest.raises(RateLimitError):
        await fl._generate_response([], None, None, ModelSize.medium)
    assert notify.exists()
    contents = notify.read_text()
    assert 'chain_exhausted' in contents
    assert 'a,b' in contents


def test_circuit_breaker_opens_after_threshold():
    """3 failures in window opens the breaker."""
    cb = _CircuitBreaker('x', threshold=3, window_sec=60)
    assert not cb.is_open()
    for _ in range(3):
        cb.record_failure(open_for_sec=10)
    assert cb.is_open()


def test_circuit_breaker_window_evicts_old_failures():
    """Failures outside the window don't count."""
    cb = _CircuitBreaker('x', threshold=3, window_sec=1)
    cb.record_failure(10)
    time.sleep(1.2)
    cb.record_failure(10)
    cb.record_failure(10)
    # Only 2 in window now → breaker still closed
    assert not cb.is_open()


def test_circuit_breaker_resets_on_success():
    """A success resets failure count."""
    cb = _CircuitBreaker('x', threshold=3, window_sec=60)
    cb.record_failure(10)
    cb.record_failure(10)
    cb.record_success()
    cb.record_failure(10)
    cb.record_failure(10)
    # Only 2 since reset → still closed
    assert not cb.is_open()


@pytest.mark.asyncio
async def test_open_breaker_skips_provider():
    """A provider with an open breaker is skipped without being called."""
    a, b = _mock_client('a'), _mock_client('b')
    fl = FailoverLLMClient([('a', a), ('b', b)])
    # Force a's breaker open
    fl.breakers['a'].open_until_ts = time.monotonic() + 60
    result = await fl._generate_response([], None, None, ModelSize.medium)
    assert result['name'] == 'b'
    a.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_resets_breaker():
    """Recording a success on the inner provider resets its breaker."""
    a, b = _mock_client('a'), _mock_client('b')
    fl = FailoverLLMClient([('a', a), ('b', b)])
    # Pre-mark some failures on 'a' but don't open
    fl.breakers['a'].record_failure(open_for_sec=10)
    fl.breakers['a'].record_failure(open_for_sec=10)
    assert len(fl.breakers['a'].failures) == 2
    await fl._generate_response([], None, None, ModelSize.medium)
    assert len(fl.breakers['a'].failures) == 0
