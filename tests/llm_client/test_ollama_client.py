"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

PATCH (donut #1211): tests for the dedicated OllamaClient.

Running tests: pytest -xvs tests/llm_client/test_ollama_client.py
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaClient,
)
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    """Pydantic schema target for structured-output tests."""

    name: str
    count: int = 0


def _mk_chat_response(payload: dict) -> SimpleNamespace:
    """Build the minimal AsyncOpenAI chat-completion response shape."""
    msg = SimpleNamespace(content=json.dumps(payload))
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


def _mk_client_with_mock(payload: dict, config: LLMConfig | None = None) -> OllamaClient:
    """Build an OllamaClient whose underlying AsyncOpenAI is a mock that
    returns `payload` as the JSON content of a chat completion."""
    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_mk_chat_response(payload)))
        )
    )
    return OllamaClient(config=config or LLMConfig(model='qwen2.5:32b'), client=fake)


def test_defaults_applied_when_config_blank():
    """A blank LLMConfig should pick up Ollama defaults."""
    fake = SimpleNamespace()  # never actually called
    client = OllamaClient(config=LLMConfig(), client=fake)
    assert client.config.base_url == DEFAULT_OLLAMA_URL
    assert client.config.api_key == 'ollama'
    assert client.model == DEFAULT_OLLAMA_MODEL


def test_explicit_config_overrides_defaults():
    fake = SimpleNamespace()
    cfg = LLMConfig(
        api_key='custom-key',
        base_url='http://other:11434/v1',
        model='qwen2.5:7b',
        temperature=0.0,
    )
    client = OllamaClient(config=cfg, client=fake)
    assert client.config.base_url == 'http://other:11434/v1'
    assert client.config.api_key == 'custom-key'
    assert client.model == 'qwen2.5:7b'


def test_caching_not_supported():
    with pytest.raises(NotImplementedError):
        OllamaClient(config=LLMConfig(), cache=True)


@pytest.mark.asyncio
async def test_generate_response_returns_parsed_json():
    payload = {'name': 'Alice', 'count': 3}
    client = _mk_client_with_mock(payload)
    msgs = [
        Message(role='system', content='you are a robot'),
        Message(role='user', content='produce JSON'),
    ]
    out = await client.generate_response(messages=msgs)
    assert out == payload


@pytest.mark.asyncio
async def test_generate_response_with_response_model_uses_json_schema():
    """When a Pydantic response_model is supplied, the request should
    carry a json_schema response_format (not just json_object)."""
    payload = {'name': 'Bob', 'count': 7}
    client = _mk_client_with_mock(payload)
    msgs = [
        Message(role='system', content='sys'),
        Message(role='user', content='go'),
    ]
    out = await client.generate_response(messages=msgs, response_model=ResponseModel)
    assert out == payload

    # Inspect the call kwargs.
    create = client.client.chat.completions.create
    create.assert_awaited()
    kwargs = create.await_args.kwargs
    assert kwargs['response_format']['type'] == 'json_schema'
    schema = kwargs['response_format']['json_schema']
    assert schema['name'] == 'ResponseModel'
    assert 'name' in schema['schema']['properties']


@pytest.mark.asyncio
async def test_generate_response_plain_uses_json_object_format():
    """No response_model => response_format should be plain json_object."""
    payload = {'name': 'C'}
    client = _mk_client_with_mock(payload)
    msgs = [Message(role='system', content='s'), Message(role='user', content='u')]
    await client.generate_response(messages=msgs)
    kwargs = client.client.chat.completions.create.await_args.kwargs
    assert kwargs['response_format'] == {'type': 'json_object'}


@pytest.mark.asyncio
async def test_passes_temperature_and_model_to_openai_sdk():
    payload = {'name': 'D'}
    cfg = LLMConfig(model='qwen2.5:32b-q4', temperature=0.0)
    client = _mk_client_with_mock(payload, config=cfg)
    msgs = [Message(role='system', content='s'), Message(role='user', content='u')]
    await client.generate_response(messages=msgs)
    kwargs = client.client.chat.completions.create.await_args.kwargs
    assert kwargs['model'] == 'qwen2.5:32b-q4'
    assert kwargs['temperature'] == 0.0
