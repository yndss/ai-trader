from typing import Any, Dict, List

import requests

from .config import get_settings


def call_llm_with_tools(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = get_settings()
    payload = {
        "model": s.openrouter_model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.2,
    }
    r = requests.post(
        f"{s.openrouter_base}/chat/completions",
        headers={
            "Authorization": f"Bearer {s.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()
    )
    r.raise_for_status()
    return r.json()
