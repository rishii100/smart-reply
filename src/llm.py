"""Minimal Groq chat client built on the Python standard library only.

Why stdlib urllib instead of the `groq` SDK or `requests`?  So the whole project
runs on a clean Python 3.14 with nothing to `pip install` but a GROQ_API_KEY.
Groq exposes an OpenAI-compatible REST API, which is trivial to call directly.

Features that matter for an evaluation harness:
  * deterministic on-disk cache keyed by the full request  -> re-runs are free
    and reproducible, and graders don't re-burn tokens;
  * retries with exponential backoff on 429/5xx;
  * a `json_object` response-format helper for structured judge output.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import config


# --------------------------------------------------------------------------- #
# .env loading (no python-dotenv dependency)
# --------------------------------------------------------------------------- #
def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite)."""
    p = Path(path) if path else config.ROOT / ".env"
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_dotenv()


class LLMError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your "
            "key (get one free at https://console.groq.com/keys)."
        )
    return key


def _cache_path(payload: dict[str, Any]) -> Path:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:24]
    return config.CACHE_DIR / f"{digest}.json"


def chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    json_mode: bool = False,
    use_cache: bool | None = None,
    cache_bust: str | None = None,
) -> str:
    """Return the assistant message content for a chat completion.

    `cache_bust` lets callers force distinct cache entries for otherwise
    identical requests (used by the judge-reliability sampler).
    """
    use_cache = config.LLM_USE_CACHE if use_cache is None else use_cache

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    cache_key = dict(body)
    if cache_bust is not None:
        cache_key = {**body, "_cache_bust": cache_bust}
    cpath = _cache_path(cache_key)
    if use_cache and cpath.exists():
        return json.loads(cpath.read_text())["content"]

    url = f"{config.GROQ_BASE_URL}/chat/completions"
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "User-Agent": "email-reply/1.0",
    }

    last_err: Exception | None = None
    for attempt in range(config.LLM_MAX_RETRIES):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            content = parsed["choices"][0]["message"]["content"]
            if use_cache:
                cpath.write_text(json.dumps({"request": body, "content": content}))
            return content
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            last_err = LLMError(f"HTTP {e.code}: {detail[:400]}")
            # Retry only on rate-limit / server errors.
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(min(2 ** attempt, 30))
                continue
            raise last_err
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = LLMError(f"network error: {e}")
            time.sleep(min(2 ** attempt, 30))
    raise last_err or LLMError("chat failed")


def chat_json(messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
    """Call `chat` in JSON mode and parse the result, tolerating stray prose.

    Falls back to non-JSON mode if the model returns a json_validate_failed error.
    """
    kwargs.setdefault("json_mode", True)
    try:
        raw = chat(messages, **kwargs)
    except LLMError as e:
        if "json_validate_failed" in str(e):
            # Retry without JSON mode — parse the raw response.
            kwargs["json_mode"] = False
            raw = chat(messages, **kwargs)
        else:
            raise
    return _loads_loose(raw)


def _loads_loose(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        # Last resort: return empty dict rather than crash the eval pipeline.
        return {}


def list_models() -> list[str]:
    """GET /models — used to verify configured model IDs exist."""
    url = f"{config.GROQ_BASE_URL}/models"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {_api_key()}", "User-Agent": "email-reply/1.0"}, method="GET"
    )
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    return sorted(m["id"] for m in parsed.get("data", []))
