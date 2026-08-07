"""Reusable optional OpenAI Responses API integration."""

from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

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
                parsed = getattr(response, "output_parsed", None)
                if parsed is None:
                    refusal = self._refusal(response)
                    detail = f": {refusal}" if refusal else ""
                    raise OpenAIExtractionError(
                        f"OpenAI returned no structured output{detail}"
                    )
                if isinstance(parsed, response_model):
                    return parsed
                return response_model.model_validate(parsed)
            except AuthenticationError as exc:
                raise OpenAIAuthenticationError(
                    "OpenAI authentication failed"
                ) from exc
            except self._TRANSIENT_ERRORS as exc:
                if attempt >= self.max_attempts:
                    raise OpenAIExtractionError(
                        f"OpenAI extraction failed after {attempt} attempts"
                    ) from exc
                self._logger.warning(
                    "Transient OpenAI failure; retrying attempt %s/%s",
                    attempt + 1,
                    self.max_attempts,
                )
                await asyncio.sleep(self.retry_delay * attempt)
            except (ValidationError, ValueError, TypeError) as exc:
                raise OpenAIExtractionError(
                    "OpenAI returned an invalid structured response"
                ) from exc
            except (APIResponseValidationError, BadRequestError) as exc:
                raise OpenAIExtractionError(
                    "OpenAI rejected or could not validate the structured response"
                ) from exc

        raise OpenAIExtractionError("OpenAI extraction failed")

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    @staticmethod
    def _refusal(response) -> str | None:
        for output in getattr(response, "output", ()) or ():
            for content in getattr(output, "content", ()) or ():
                refusal = getattr(content, "refusal", None)
                if refusal:
                    return str(refusal)
        return None
