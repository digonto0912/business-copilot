"""Per-model Gemini RPM/TPM limiter.

The limiter runs before each actual chat-model request. It uses the fixed
UTC-08:00 minute window requested for Gemini quotas and reserves:

    estimated input tokens + 3,000 max output tokens + 5,000 safety tokens

The reservation prevents a request from being sent when its worst-case
budget would cross the configured TPM limit. After the call completes, the
reservation is reconciled with the actual token usage reported by the model.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


# User-provided model limits.
MODEL_LIMITS = {
    "gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250_000},
    "gemma-4-31b-it": {"rpm": 30, "tpm": 16_000},
}

MAX_OUTPUT_TOKENS_FOR_RESERVATION = 3_000
SAFETY_TOKEN_MARGIN = 5_000

# Google quota window requested by the user: fixed UTC-08, not local time.
UTC_MINUS_8 = timezone(timedelta(hours=-8))


class _QuotaState:
    def __init__(self) -> None:
        self.window_key: str | None = None
        self.requests = 0
        self.tokens_used = 0
        self.tokens_in_flight = 0
        self.lock = threading.RLock()

    def reset_if_needed(self, window_key: str) -> None:
        if self.window_key != window_key:
            self.window_key = window_key
            self.requests = 0
            self.tokens_used = 0
            self.tokens_in_flight = 0


_STATES: dict[str, _QuotaState] = defaultdict(_QuotaState)


def _window_info(now: datetime | None = None) -> tuple[str, float]:
    """Return (minute-key, seconds-until-next-minute) in fixed UTC-08."""
    current = now.astimezone(UTC_MINUS_8) if now else datetime.now(UTC_MINUS_8)
    key = current.strftime("%Y-%m-%d %H:%M")
    next_minute = (current.replace(second=0, microsecond=0) + timedelta(minutes=1))
    wait_seconds = max(0.0, (next_minute - current).total_seconds())
    return key, wait_seconds


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """Best-effort extraction of input/output/total token usage."""
    usage = None

    # Modern LangChain AIMessage usage metadata.
    try:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
    except Exception:
        pass

    if usage:
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        return input_tokens, output_tokens, total_tokens

    # Provider callback metadata.
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
        output_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens"))
        total_tokens = token_usage.get("total_tokens")
        return input_tokens, output_tokens, total_tokens
    except Exception:
        return None, None, None


def _estimate_tokens(model: Any, messages: Any) -> int:
    """Estimate prompt tokens using the model tokenizer when available."""
    try:
        return int(model.get_num_tokens_from_messages(messages))
    except Exception:
        # Conservative fallback: roughly 4 UTF-8-ish characters per token.
        try:
            if messages and hasattr(messages[0], "content"):
                flat_messages = messages
            else:
                flat_messages = [message for batch in messages for message in batch]
            text = "\n".join(str(getattr(message, "content", "")) for message in flat_messages)
        except Exception:
            text = str(messages)
        return max(1, (len(text) + 3) // 4)


class GeminiQuotaManager:
    """Thread-safe per-model RPM/TPM gate and usage tracker."""

    def before_request(self, model_name: str, estimated_input_tokens: int) -> int:
        limits = MODEL_LIMITS.get(model_name)
        if limits is None:
            return 0

        state = _STATES[model_name]
        request_budget = max(0, int(estimated_input_tokens)) + MAX_OUTPUT_TOKENS_FOR_RESERVATION + SAFETY_TOKEN_MARGIN

        if request_budget >= limits["tpm"]:
            raise RuntimeError(
                f"The estimated token budget for one {model_name} request "
                f"({request_budget:,} tokens) is too large for its {limits["tpm"]:,} TPM limit."
            )

        while True:
            key, wait_seconds = _window_info()
            with state.lock:
                state.reset_if_needed(key)

                requests_left = limits["rpm"] - state.requests
                tokens_left = limits["tpm"] - (state.tokens_used + state.tokens_in_flight)

                # Never send the request if it would consume the final RPM slot,
                # or if the reserved worst-case token budget would cross TPM.
                rpm_safe = requests_left > 1
                tpm_safe = request_budget < tokens_left

                if rpm_safe and tpm_safe:
                    state.requests += 1
                    state.tokens_in_flight += request_budget
                    return request_budget

            # Wait for the next UTC-08 minute boundary before retrying.
            time.sleep(max(0.05, wait_seconds))
    def snapshot(self, model_name: str) -> dict[str, Any]:
        limits = MODEL_LIMITS.get(model_name)
        if limits is None:
            return {}

        key, _ = _window_info()
        state = _STATES[model_name]
        with state.lock:
            state.reset_if_needed(key)
            return {
                "model": model_name,
                "window_utc_minus_8": key,
                "rpm_limit": limits["rpm"],
                "rpm_used": state.requests,
                "rpm_remaining": max(0, limits["rpm"] - state.requests),
                "tpm_limit": limits["tpm"],
                "tpm_used": state.tokens_used,
                "tpm_in_flight_reserved": state.tokens_in_flight,
                "tpm_remaining": max(0, limits["tpm"] - state.tokens_used - state.tokens_in_flight),
                "max_output_tokens_reserved": MAX_OUTPUT_TOKENS_FOR_RESERVATION,
                "safety_token_margin": SAFETY_TOKEN_MARGIN,
            }


    def reconcile(
        self,
        model_name: str,
        reserved_tokens: int,
        actual_input_tokens: int | None,
        actual_output_tokens: int | None,
        actual_total_tokens: int | None,
    ) -> None:
        if model_name not in MODEL_LIMITS:
            return

        if actual_total_tokens is None:
            if actual_input_tokens is not None or actual_output_tokens is not None:
                actual_total_tokens = (actual_input_tokens or 0) + (actual_output_tokens or 0)
            else:
                actual_total_tokens = reserved_tokens

        state = _STATES[model_name]
        key, _ = _window_info()
        with state.lock:
            state.reset_if_needed(key)
            state.tokens_in_flight = max(0, state.tokens_in_flight - reserved_tokens)
            state.tokens_used += int(actual_total_tokens)

    def release_on_error(self, model_name: str, reserved_tokens: int) -> None:
        if model_name not in MODEL_LIMITS:
            return
        state = _STATES[model_name]
        key, _ = _window_info()
        with state.lock:
            state.reset_if_needed(key)
            # Keep the reservation on errors. The request has already started, so
            # retaining the budget avoids under-counting TPM usage.
            return


quota_manager = GeminiQuotaManager()


class GeminiQuotaCallback(BaseCallbackHandler):
    """Gate one concrete chat-model instance before it sends a request."""

    def __init__(self, model_name: str, model: Any) -> None:
        self.model_name = model_name
        self.model = model
        self._reservations: dict[str, int] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        # Ignore non-rate-limited models.
        if self.model_name not in MODEL_LIMITS:
            return

        estimated_input = _estimate_tokens(self.model, messages)
        reserved = quota_manager.before_request(self.model_name, estimated_input)
        self._reservations[str(run_id)] = reserved

    def on_llm_end(self, response, *, run_id, **kwargs):
        reservation = self._reservations.pop(str(run_id), None)
        if reservation is None:
            return

        input_tokens, output_tokens, total_tokens = _extract_usage(response)
        quota_manager.reconcile(
            self.model_name,
            reservation,
            input_tokens,
            output_tokens,
            total_tokens,
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        reservation = self._reservations.pop(str(run_id), None)
        if reservation is not None:
            quota_manager.release_on_error(self.model_name, reservation)


# One callback object per actual Google model instance.
# The callback is reused safely across calls because reservations are keyed by run_id.

gemini_31_flash_lite_quota = GeminiQuotaCallback("gemini-3.1-flash-lite", None)
gemma_4_31b_quota = GeminiQuotaCallback("gemma-4-31b-it", None)


def get_quota_snapshot(model_name: str) -> dict[str, Any]:
    return quota_manager.snapshot(model_name)
