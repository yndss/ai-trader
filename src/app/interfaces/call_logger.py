"""Simple in-memory logger for tracking MCP tool calls per question."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from typing import Any, Dict, List


class CallLogger:
    """Stores tool call history keyed by the current user question."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._current_question: str | None = None
        self._current_token: str | None = None

    def clear_question_history(self, question: str) -> None:
        with self._lock:
            self._history.pop(question, None)

    def set_current_question(self, question: str) -> str:
        token = str(uuid.uuid4())
        with self._lock:
            self._current_question = question
            self._current_token = token
            self._history.setdefault(question, [])
        return token

    def reset_current_question(self, token: str) -> None:
        with self._lock:
            if self._current_token == token:
                self._current_question = None
                self._current_token = None

    def log_tool_call(self, tool_name: str, params: Dict[str, Any]) -> None:
        with self._lock:
            if self._current_question is None:
                return
            sanitized = {}
            sensitive_keys = {"secret", "token", "jwt", "authorization", "password"}
            for key, value in params.items():
                key_lower = key.lower() if isinstance(key, str) else ""
                sanitized[key] = "***" if key_lower in sensitive_keys else value
            self._history[self._current_question].append({"tool": tool_name, "params": sanitized})

    def question_history(self, question: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history.get(question, []))


call_logger = CallLogger()
