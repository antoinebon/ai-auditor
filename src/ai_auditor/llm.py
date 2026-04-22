"""LangChain chat-model factory + JSON-structured-output helper.

The pipeline depends on ``BaseChatModel.invoke`` + ``BaseChatModel.bind_tools``
and nothing more — any modern tool-calling-capable LangChain chat model works
as a drop-in. ``make_llm`` is the one seam where the concrete provider is
chosen; today it returns a ``ChatOllama`` because that's the project's default,
but the return type is ``BaseChatModel`` so callers don't care.

``call_json`` wraps a single structured-output call with pydantic validation
and a one-shot retry that quotes the validation error back to the model —
small local models sometimes emit a trailing sentence or an extra key, and
one retry clears most of those cases without thrashing token budgets.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from ai_auditor.config import Settings

logger = logging.getLogger(__name__)


def make_llm(
    settings: Settings,
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Return the project's configured chat model.

    Currently returns a ``ChatOllama`` — swap the body to change providers.
    ``json_mode=True`` maps to Ollama's ``format="json"``; other providers
    would need their own mapping (``response_format={"type": "json_object"}``
    for OpenAI, tool-calling fallback for Anthropic, etc.). Return type is
    ``BaseChatModel`` so callers see only the provider-neutral interface.
    """
    return ChatOllama(
        base_url=settings.ollama_host,
        model=settings.ollama_model,
        temperature=temperature,
        format="json" if json_mode else None,
    )


def call_json[ModelT: BaseModel](
    llm: BaseChatModel,
    system: str,
    user: str,
    schema: type[ModelT],
) -> ModelT:
    """Call ``llm`` with a (system, user) prompt pair and parse to ``schema``.

    The model is expected to be configured for structured output (JSON mode
    for providers that offer it). On a single ``ValidationError`` we send
    the same prompt again with the validation error appended so the model
    can self-correct. A second failure raises.
    """
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]
    response = llm.invoke(messages)
    raw = content_text(response)
    try:
        return schema.model_validate_json(raw)
    except ValidationError as first_error:
        logger.warning(
            "LLM structured output failed schema validation on first attempt; retrying. error=%s",
            first_error,
        )
        retry_user = (
            f"{user}\n\n"
            "Your previous response did not match the required schema. "
            f"Validation error:\n{first_error}\n"
            "Respond again with a single JSON object that matches the schema. "
            "Do not include prose before or after the JSON."
        )
        messages = [SystemMessage(content=system), HumanMessage(content=retry_user)]
        retry_response = llm.invoke(messages)
        retry_raw = content_text(retry_response)
        return schema.model_validate_json(retry_raw)


def content_text(message: AIMessage | Any) -> str:
    """Extract plain text from a LangChain message.

    Providers return content as either a string (OpenAI, Ollama usually) or
    a list of content-blocks (Anthropic, especially alongside tool calls).
    We accept both and serialise the list form back to JSON text.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    # content is a list of dicts / strings; join text parts.
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append(json.dumps(part))
    return "".join(parts)
