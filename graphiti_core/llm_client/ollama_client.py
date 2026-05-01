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

PATCH (donut #1211): dedicated Ollama client.

Ollama serves an OpenAI-compatible API but does NOT implement the
`responses.parse` Pydantic-structured-output endpoint that
`OpenAIClient` uses. This client uses plain `chat.completions.create`
with `response_format={"type": "json_object"}` (or json_schema when a
Pydantic response_model is supplied) and post-parses the JSON into the
target schema.

Pairs with the V3 entity-extraction prompt
(`graphiti_core/prompts/extract_nodes.py`, gated on env
`GRAPHITI_USE_V3_PROMPT=1`) so a graphiti instance dedicated to local
extraction can run a separate prompt suited to Qwen 2.5 32B.
"""

import logging
import typing

from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import LLMConfig
from .openai_generic_client import OpenAIGenericClient

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = 'http://localhost:11434/v1'
DEFAULT_OLLAMA_MODEL = 'qwen2.5:32b-instruct-q4_K_M'
# Local Qwen 32B Q4 on CPU is much slower than cloud APIs; 60-100KB
# episode_body inputs took 4-10 minutes in early benches. The graphiti
# default openai timeout is 600s; bump higher for local.
DEFAULT_OLLAMA_TIMEOUT_S = 1800.0
# Local models benefit from more headroom than the 8K cloud default.
DEFAULT_OLLAMA_MAX_TOKENS = 16384


class OllamaClient(OpenAIGenericClient):
    """LLM client targeting a local Ollama OpenAI-compatible server.

    Inherits the JSON / json_schema chat-completions path from
    `OpenAIGenericClient`. The behavioral difference vs. the upstream
    `OpenAIGenericClient` is:

      - Defaults aimed at Ollama (`base_url`, `model`, big `max_tokens`,
        long request timeout).
      - Provider tag in tracing reads `ollama` for log clarity.

    Use the V3 prompt by setting `GRAPHITI_USE_V3_PROMPT=1` in the env
    of the graphiti server process that owns this client.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_OLLAMA_MAX_TOKENS,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT_S,
    ):
        if cache:
            raise NotImplementedError('Caching is not implemented for Ollama')

        if config is None:
            config = LLMConfig()

        # Default to local Ollama OpenAI-compat endpoint when not set.
        base_url = config.base_url or DEFAULT_OLLAMA_URL
        api_key = config.api_key or 'ollama'

        if client is None:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        # OpenAIGenericClient fully supports the json_schema / json_object
        # path we want; reuse its __init__ but replace base_url/api_key.
        config.base_url = base_url
        config.api_key = api_key
        if not config.model:
            config.model = DEFAULT_OLLAMA_MODEL

        super().__init__(config=config, cache=False, client=client, max_tokens=max_tokens)
        self.timeout = timeout

    async def generate_response(
        self,
        messages,
        response_model: type[BaseModel] | None = None,
        max_tokens=None,
        model_size=None,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ):
        # Re-tag the tracing span so logs show ollama vs openai. Easiest
        # way is to delegate to parent and also log here for grep-ability.
        if prompt_name:
            logger.debug(
                'OllamaClient.generate_response model=%s prompt=%s group=%s',
                self.model, prompt_name, group_id,
            )
        # OpenAIGenericClient accepts ModelSize default; pass through.
        from .config import ModelSize  # local import to avoid cycle issues
        if model_size is None:
            model_size = ModelSize.medium
        return await super().generate_response(
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
        )
