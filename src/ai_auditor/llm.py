"""Thin LangChain Ollama wrapper + JSON-structured-output helper.

Keeps the rest of the code free of LangChain messaging types by exposing a
single ``call_json`` that takes a system prompt + user prompt and returns a
validated pydantic model. On a single ValidationError we retry once with a
correction message that quotes the validation error — small local models
sometimes emit a trailing sentence or an extra key, and one retry clears
most of those cases without thrashing token budgets.
"""

from __future__ import annotations

import json
import logging
from typing import Any

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
) -> ChatOllama:
    """Return a configured ``ChatOllama`` instance.

    ``json_mode=True`` sets Ollama's ``format`` parameter to ``"json"`` so the
    model emits a syntactically valid JSON object. It does **not** guarantee
    the object matches a given schema — that's what ``call_json`` adds.
    """
    return ChatOllama(
        base_url=settings.ollama_host,
        model=settings.ollama_model,
        temperature=temperature,
        format="json" if json_mode else None,
    )


def call_json[ModelT: BaseModel](
    llm: ChatOllama,
    system: str,
    user: str,
    schema: type[ModelT],
) -> ModelT:
    """Call ``llm`` with a (system, user) prompt pair and parse to ``schema``.

    The model is expected to be configured with ``json_mode=True``. On a
    single ``ValidationError`` we send the same prompt again with the
    validation error appended so the model can self-correct. A second
    failure raises.
    """
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]
    response: AIMessage = llm.invoke(messages)
    raw = _content_text(response)
    try:
        return schema.model_validate_json(raw)
    except ValidationError as first_error:
        logger.warning(
            "LLM structured output failed schema validation on first attempt; retrying. "
            "error=%s",
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
        retry_response: AIMessage = llm.invoke(messages)
        retry_raw = _content_text(retry_response)
        return schema.model_validate_json(retry_raw)


def _content_text(message: AIMessage) -> str:
    """Extract plain text from an ``AIMessage``.

    LangChain can return either a string or a list of content parts. We
    accept both and serialise the list form back to JSON text.
    """
    content = message.content
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
