"""Tiny stdlib JSON-over-HTTP helper with bounded retries."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from llm.types import ChatError

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout: float = 120.0,
    max_retries: int = 3,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    last: str = ""
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500] if hasattr(exc, "read") else ""
            last = f"HTTP {exc.code}: {detail}"
            if exc.code not in RETRYABLE_STATUS:
                raise ChatError(last) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = f"network error: {exc}"
        except json.JSONDecodeError as exc:
            raise ChatError(f"response was not JSON: {exc}") from exc
        if attempt < max_retries:
            (sleep or time.sleep)(min(2.0**attempt, 20.0))
    raise ChatError(f"giving up after {max_retries + 1} attempts; last error: {last}")
