#!/usr/bin/env python3
"""Streamlit веб-клиент поверх MCP-оркестратора."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import streamlit as st

from src.app.interfaces.call_logger import call_logger
from src.app.interfaces.mcp_streamlit_service import MCPOrchestratorService


DEFAULT_BASE_URL = os.getenv("FINAM_API_BASE_URL", "https://api.finam.ru")


def _env_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _ensure_state_defaults() -> None:
    """Устанавливает значения по умолчанию в состоянии сессии."""
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "finam_token" not in st.session_state:
        st.session_state.finam_token = os.getenv("FINAM_ACCESS_TOKEN", "")
    if "finam_base_url" not in st.session_state:
        st.session_state.finam_base_url = DEFAULT_BASE_URL
    if "account_id" not in st.session_state:
        st.session_state.account_id = os.getenv("DEFAULT_ACCOUNT_ID", "")
    if "_initial_defaults" not in st.session_state:
        st.session_state._initial_defaults = {
            "account_id": os.getenv("DEFAULT_ACCOUNT_ID", ""),
            "account_id_alt": os.getenv("DEFAULT_ACCOUNT_ID", ""),
        }


def _reset_service() -> None:
    service = st.session_state.pop("mcp_service", None)
    if service is not None:
        try:
            service.close()
        except Exception as exc:  # pragma: no cover - best effort logging
            st.sidebar.warning(f"⚠️ Не удалось корректно остановить MCP: {exc}")
    st.session_state.pop("mcp_service_config", None)


def _apply_account_defaults(account_id: str) -> str:
    initial_account = st.session_state._initial_defaults["account_id"]
    account_for_use = account_id or initial_account
    # Store the account ID in environment for use by the MCP server
    os.environ["DEFAULT_ACCOUNT_ID"] = account_for_use
    return account_for_use


def _service_config() -> Tuple[str, str, str]:
    token = (st.session_state.finam_token or "").strip()
    base_url = (st.session_state.finam_base_url or DEFAULT_BASE_URL).strip()
    account_id = (st.session_state.account_id or "").strip()
    return token, base_url, account_id


def _get_service() -> MCPOrchestratorService:
    token, base_url, account_id = _service_config()
    current_config = st.session_state.get("mcp_service_config")
    service: MCPOrchestratorService | None = st.session_state.get("mcp_service")

    if current_config != (token, base_url, account_id) and service is not None:
        _reset_service()
        service = None

    if service is None:
        account_for_use = _apply_account_defaults(account_id)

        if token:
            os.environ["FINAM_ACCESS_TOKEN"] = token
        elif "FINAM_ACCESS_TOKEN" in os.environ:
            os.environ.pop("FINAM_ACCESS_TOKEN")

        os.environ["FINAM_API_BASE_URL"] = base_url or DEFAULT_BASE_URL
        os.environ["DEFAULT_ACCOUNT_ID"] = account_for_use

        service = MCPOrchestratorService()
        st.session_state.mcp_service = service
        st.session_state.mcp_service_config = (token, base_url, account_id)

    return service


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            tool_calls: List[Dict[str, Any]] = message.get("tool_calls", [])  # type: ignore[assignment]
            if tool_calls:
                with st.expander("🔧 Вызовы MCP инструментов", expanded=False):
                    for idx, call in enumerate(tool_calls, start=1):
                        st.markdown(f"**#{idx}** {call['tool']}")
                        st.json(call.get("params", {}))


def main() -> None:  # noqa: C901
    st.set_page_config(page_title="AI Трейдер (Finam)", page_icon="🤖", layout="wide")
    _ensure_state_defaults()

    st.title("🤖 AI Ассистент Трейдера")
    st.caption("Интерфейс поверх MCP-оркестратора Finam TradeAPI")

    with st.sidebar:
        st.header("⚙️ Настройки")

        model_name = _env_value("OPENROUTER_MODEL", "COMET_MODEL_ID", "LLM_MODEL_ID") or "openai/gpt-4o-mini"
        st.info(f"**Модель:** {model_name}")

        with st.expander("🔑 Finam API", expanded=False):
            st.session_state.finam_token = st.text_input(
                "Access Token",
                value=st.session_state.finam_token,
                type="password",
                help="Токен доступа к Finam TradeAPI (FINAM_ACCESS_TOKEN)",
            )
            st.session_state.finam_base_url = st.text_input(
                "API Base URL",
                value=st.session_state.finam_base_url,
                help="Базовый URL API",
            )

        st.session_state.account_id = st.text_input(
            "ID счёта",
            value=st.session_state.account_id,
            help="Используется, если вопрос не содержит явно account_id",
        )

        if st.button("🔄 Очистить историю"):
            st.session_state.messages = []
            _reset_service()
            st.rerun()

        st.markdown("---")
        st.markdown("### 💡 Примеры вопросов:")
        st.markdown(
            """
        - Авторизуйся с моим токеном
        - Покажи баланс и позиции
        - Какие последние сделки по Сберу?
        - Построй стакан по Газпрому
        - Создай рыночный ордер на покупку
        """
        )

        api_key_present = bool(_env_value("OPENROUTER_API_KEY", "COMET_API_KEY", "LLM_API_KEY"))
        if api_key_present:
            st.success("✅ Ключ OpenRouter установлен")
        else:
            st.warning("⚠️ Укажите OPENROUTER_API_KEY (или COMET_API_KEY/LLM_API_KEY)")

    _render_history()

    prompt = st.chat_input("Напишите ваш вопрос...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                service = _get_service()
                response_text = service.process_request(prompt)
                tool_calls = call_logger.question_history(prompt)

                st.markdown(response_text)

                if tool_calls:
                    with st.expander("🔧 Вызовы MCP инструментов", expanded=False):
                        for idx, call in enumerate(tool_calls, start=1):
                            st.markdown(f"**#{idx}** {call['tool']}")
                            st.json(call.get("params", {}))

                message_data: Dict[str, Any] = {"role": "assistant", "content": response_text}
                if tool_calls:
                    message_data["tool_calls"] = tool_calls
                st.session_state.messages.append(message_data)
            except Exception as exc:
                st.error(f"❌ Ошибка: {exc}")


if __name__ == "__main__":
    main()
