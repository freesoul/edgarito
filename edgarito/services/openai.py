"""Reusable optional OpenAI Responses API integration."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from edgarito.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
)

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class OpenAIUnavailableError(RuntimeError):
    """Raised when optional OpenAI extraction is not configured."""


class OpenAIExtractionError(RuntimeError):
    """Raised for a refused, malformed, or failed structured extraction."""


class OpenAIAuthenticationError(OpenAIExtractionError):
    """Raised for a non-retryable credential failure."""


class OpenAIClient:
    """Small async wrapper around Responses API Structured Outputs.

    The wrapper deliberately has no financial-domain behavior. Callers provide
    their own instructions and Pydantic response model.
    """

    _TRANSIENT_ERRORS = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        reasoning_effort: str = OPENAI_REASONING_EFFORT,
        client: AsyncOpenAI | None = None,
        max_attempts: int = 3,
        retry_delay: float = 0.25,
        timeout: float = 60,
    ) -> None:
        self._logger = logging.getLogger(type(self).__name__)
        self.api_key_configured = bool(api_key or client is not None)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_attempts = max(1, max_attempts)
        self.retry_delay = max(0, retry_delay)
        self.timeout = timeout
        self._client = client
        if self._client is None and api_key:
            # Disable SDK retries so the bounded policy below is the only retry
            # layer and remains straightforward to test and audit.
            self._client = AsyncOpenAI(api_key=api_key, max_retries=0)

    async def extract_structured(
        self,
        *,
        instructions: str,
        content: str,
        response_model: type[StructuredModel],
        model: str | None = None,
    ) -> StructuredModel:
        if self._client is None:
            raise OpenAIUnavailableError("OpenAI is not configured")

        selected_model = model or self.model
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._client.responses.parse(
                    model=selected_model,
                    instructions=instructions,
                    input=content,
                    text_format=response_model,
                    reasoning={"effort": self.reasoning_effort},
                    store=False,
                    timeout=self.timeout,
                )
                parsed = self._structured_output(response)
                if parsed is None:
                    refusal = self._refusal(response)
                    reason = refusal or self._response_reason(response)
                    detail = f": {reason}" if reason else ""
                    raise OpenAIExtractionError(
                        f"OpenAI returned no structured output{detail}"
                    )
                if isinstance(parsed, response_model):
                    return parsed
                return response_model.model_validate(parsed)
            except AuthenticationError as exc:
                raise OpenAIAuthenticationError("OpenAI authentication failed") from exc
            except self._TRANSIENT_ERRORS as exc:
                if attempt >= self.max_attempts:
                    raise OpenAIExtractionError(
                        "OpenAI extraction failed after "
                        f"{attempt} attempts: {self._exception_detail(exc)}"
                    ) from exc
                self._logger.warning(
                    "Transient OpenAI failure; retrying attempt %s/%s",
                    attempt + 1,
                    self.max_attempts,
                )
                await asyncio.sleep(self.retry_delay * attempt)
            except (ValidationError, ValueError, TypeError) as exc:
                raise OpenAIExtractionError(
                    "OpenAI returned an invalid structured response: "
                    f"{self._exception_detail(exc)}"
                ) from exc
            except (APIResponseValidationError, BadRequestError) as exc:
                raise OpenAIExtractionError(
                    "OpenAI rejected or could not validate the structured response: "
                    f"{self._exception_detail(exc)}"
                ) from exc

        raise OpenAIExtractionError("OpenAI extraction failed")

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    @staticmethod
    def _refusal(response) -> str | None:
        for output in OpenAIClient._values(OpenAIClient._field(response, "output")):
            for content in OpenAIClient._values(OpenAIClient._field(output, "content")):
                refusal = OpenAIClient._field(content, "refusal")
                if refusal:
                    return str(refusal)
        return None

    @classmethod
    def _structured_output(cls, response: Any) -> Any | None:
        """Read parsed Responses output across supported SDK response shapes.

        Current ``responses.parse`` exposes ``output_parsed``.  The underlying
        response shape, and compatible SDKs/proxies, expose the same value as
        ``output[*].content[*].parsed`` instead.  A final JSON-text fallback is
        deliberately limited to those output-text/content fields; it does not
        recursively search arbitrary response data or accept forecast fields.
        The response model remains responsible for the strict evidence-only
        validation after this transport normalization.
        """

        parsed = cls._field(response, "output_parsed")
        if parsed is not None:
            return parsed

        for output in cls._values(cls._field(response, "output")):
            parsed = cls._field(output, "parsed")
            if parsed is not None:
                return parsed
            for content in cls._values(cls._field(output, "content")):
                parsed = cls._field(content, "parsed")
                if parsed is not None:
                    return parsed
                parsed = cls._json_object(cls._field(content, "text"))
                if parsed is not None:
                    return parsed

        # Keep compatibility with callers that supply a Chat Completions-like
        # fake while the production client uses Responses.
        for choice in cls._values(cls._field(response, "choices")):
            message = cls._field(choice, "message")
            parsed = cls._field(message, "parsed")
            if parsed is not None:
                return parsed
            parsed = cls._json_object(cls._field(message, "content"))
            if parsed is not None:
                return parsed

        return cls._json_object(cls._field(response, "output_text"))

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _values(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, Mapping):
            return (value,)
        if isinstance(value, (str, bytes)):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    @staticmethod
    def _json_object(value: Any) -> Any | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    @classmethod
    def _response_reason(cls, response: Any) -> str | None:
        """Return status/error details when a response has no parsed output."""

        error = cls._field(response, "error")
        error_message = cls._field(error, "message")
        if error_message:
            return str(error_message)
        incomplete = cls._field(response, "incomplete_details")
        incomplete_reason = cls._field(incomplete, "reason")
        if incomplete_reason:
            return f"incomplete response ({incomplete_reason})"
        status = cls._field(response, "status")
        return f"response status={status}" if status and status != "completed" else None

    @staticmethod
    def _exception_detail(exc: Exception) -> str:
        """Return a concise provider/Pydantic reason suitable for audit output."""

        body = getattr(exc, "body", None)
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping) and error.get("message"):
                return str(error["message"])
            if body.get("message"):
                return str(body["message"])
        detail = str(exc).strip()
        return detail or type(exc).__name__
