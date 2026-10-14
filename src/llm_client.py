from __future__ import annotations

import os


import json
import urllib.error
import urllib.request
from typing import Dict

API_KEY = os.environ.get("PROJECTION_API_KEY", "")
BASE_URL = "https://automl.aiserverai.online/v1"
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def chat_completions(payload: Dict, *, timeout: int = 1500, api_key: str = "", base_url: str = "") -> Dict:
    """Call OpenAI-compatible /chat/completions endpoint and return parsed JSON."""
    key = (api_key or API_KEY).strip()
    url_base = (base_url or BASE_URL).rstrip("/")
    if not key:
        raise RuntimeError("API key is empty.")

    payload = dict(payload)
    payload["stream"] = False

    req = urllib.request.Request(
        url=f"{url_base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {e.code}: {detail}") from e
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}") from e

    return json.loads(body)
