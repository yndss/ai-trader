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
    if "sidebar_state" not in st.session_state:
        st.session_state.sidebar_state = "expanded"


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
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #3B82F6, #1E3A8A); color: #FFFFFF; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);">
                    {content}
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #3B82F6, #1E3A8A); color: #FFFFFF; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; border-left: 4px solid #10B981; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);">
                    {content}
                </div>
                """, unsafe_allow_html=True)

                tool_calls: List[Dict[str, Any]] = message.get("tool_calls", [])  # type: ignore[assignment]
                if tool_calls:
                    with st.expander("🔧 Детали выполнения MCP инструментов", expanded=False):
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #FEF3C7, #FDE68A); color: #92400E; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; font-weight: 600;">
                            🛠️ Выполненные операции:
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for idx, call in enumerate(tool_calls, start=1):
                            st.markdown(f"""
                            <div style="background: #FFFFFF; color: #1F2937; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #10B981; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);">
                                <strong>#{idx} {call['tool']}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if call.get("params"):
                                st.json(call["params"])


def main() -> None:  # noqa: C901
    st.set_page_config(page_title="AI Трейдер (Finam)", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    _ensure_state_defaults()
    
    # Custom CSS styling
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1E3A8A;
        --secondary-color: #3B82F6;
        --accent-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style the bottom block container like sidebar */
    .stBottomBlockContainer {
        background: linear-gradient(180deg, #1E3A8A 0%, #3B82F6 100%) !important;
        border-radius: 15px 15px 0 0 !important;
        margin: 1rem !important;
        padding: 1rem !important;
        color: white !important;
    }
    
    .stBottomBlockContainer > div {
        background: transparent !important;
    }
    
    .stBottomBlockContainer * {
        color: white !important;
    }
    
    /* Hide the original sidebar collapse button */
    .stSidebarCollapseButton {
        display: none !important;
    }
    
    /* Hide all possible sidebar collapse/toggle buttons */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    [data-testid="stSidebarNav"] button {
        display: none !important;
    }
    
    .css-1rs6os button {
        display: none !important;
    }
    
    .css-17lntkn button {
        display: none !important;
    }
    
    /* Hide any button in sidebar header area */
    .stSidebar header button {
        display: none !important;
    }
    
    .stSidebar [kind="header"] button {
        display: none !important;
    }
    
    /* Force sidebar to stay open */
    .stSidebar {
        min-width: 300px !important;
        transform: translateX(0px) !important;
    }

    .stBottom {
        background-color: #262730;
    }
    
    /* Custom background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Title styling */
    h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 15px;
        margin: 1rem;
        padding: 1rem;
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    /* Sidebar headers */
    .sidebar h2, .sidebar h3 {
        color: white !important;
        font-weight: 600;
    }
    
    /* Text input root element - make it NOT white */
    .stTextInputRootElement {
        background: transparent !important;
    }
    
    .stTextInputRootElement > div {
        background: transparent !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #3B82F6, #10B981);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        height: 40px;
        min-height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    
    /* Ensure consistent button heights in columns */
    .stColumn .stButton > button {
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        line-height: 1 !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 0.75rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-left: 4px solid #3B82F6;
        color: #FFFFFF !important;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10B981;
        color: #FFFFFF !important;
    }
    
    /* Ensure all text in chat messages is properly colored */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: inherit !important;
    }
    
    /* Markdown content in messages */
    .stMarkdown p {
        color: inherit !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #F3F4F6, #E5E7EB);
        border-radius: 10px;
        font-weight: 600;
        color: #1F2937 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 0 0 10px 10px;
        color: #1F2937 !important;
    }
    
    /* Ensure all expander content is dark text */
    .streamlit-expanderContent p, 
    .streamlit-expanderContent div, 
    .streamlit-expanderContent span,
    .streamlit-expanderContent label {
        color: #1F2937 !important;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        border-radius: 10px;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        border-radius: 10px;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        border-radius: 10px;
        border: none;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #3B82F6, #1E3A8A);
        color: white;
        border-radius: 10px;
        border: none;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #3B82F6 !important;
    }
    
    /* Caption styling */
    .css-1v0mbdj {
        color: #FFFFFF !important;
        font-style: italic;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Global text color fixes */
    .main p, .main div, .main span {
        color: #1F2937;
    }
    
    /* Sidebar text should be white */
    .sidebar p, .sidebar div, .sidebar span, .sidebar label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Input labels in sidebar */
    .sidebar .stTextInput label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Help text */
    .sidebar .help {
        color: rgba(255, 255, 255, 0.7) !important;
    }
                
    [data-testid="stBottomBlockContainer"] {
        background-color: #222222;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AI Ассистент Трейдера")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #FFFFFF; font-weight: 500; margin-top: -1rem; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);">
            Интеллектуальный помощник для работы с Finam TradeAPI
        </p>
        <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin-top: 1rem;">
            <span style="background: linear-gradient(45deg, #047857, #065F46); color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; font-weight: 600;">Powered by MCP</span>
            <span style="background: linear-gradient(45deg, #1E40AF, #1E3A8A); color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; font-weight: 600;">AI Enhanced</span>
            <span style="background: linear-gradient(45deg, #D97706, #B45309); color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; font-weight: 600;">Multi-agent</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h2 style="color: white; margin-bottom: 0.5rem;">⚙️ Панель управления</h2>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Настройки и конфигурация</p>
        </div>
        """, unsafe_allow_html=True)

        model_name = _env_value("OPENROUTER_MODEL", "COMET_MODEL_ID", "LLM_MODEL_ID") or "openai/gpt-4o-mini"
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="color: white; font-weight: 600; margin-bottom: 0.5rem;">🧠 Модель ИИ</div>
            <div style="color: rgba(255, 255, 255, 0.9); font-size: 0.9rem; font-family: monospace;">{model_name}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔑 Настройки Finam API", expanded=False):
            st.markdown('<div style="color: #1F2937; font-weight: 600;">**Конфигурация подключения к Finam TradeAPI**</div>', unsafe_allow_html=True)
            st.session_state.finam_token = st.text_input(
                "🔐 Access Token",
                value=st.session_state.finam_token,
                type="password",
                help="Токен доступа к Finam TradeAPI (FINAM_ACCESS_TOKEN)",
                placeholder="Введите ваш токен доступа..."
            )
            st.session_state.finam_base_url = st.text_input(
                "🌐 API Base URL",
                value=st.session_state.finam_base_url,
                help="Базовый URL API",
                placeholder="https://api.finam.ru"
            )

        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="color: white; font-weight: 600; margin-bottom: 0.5rem;">💼 Настройки счета</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.account_id = st.text_input(
            "🏦 ID счёта",
            value=st.session_state.account_id,
            help="Используется, если вопрос не содержит явно account_id",
            placeholder="Введите ID вашего счета..."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Очистить", help="Очистить историю чата", key="clear_btn", use_container_width=True):
                st.session_state.messages = []
                _reset_service()
                st.rerun()
        
        with col2:
            if st.button("📊 Статус", help="Проверить подключение", key="status_btn", use_container_width=True):
                st.info("🔄 Проверка соединения...")

        st.markdown("---")
        
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="color: white; font-weight: 600; margin-bottom: 1rem;">💡 Примеры запросов</div>
            <div style="color: rgba(255, 255, 255, 0.9); font-size: 0.85rem; line-height: 1.5;">
                🔐 Авторизуйся с моим токеном<br>
                💰 Покажи баланс и позиции<br>
                📈 Последние сделки по Сберу<br>
                📊 Построй стакан по Газпрому<br>
                🛒 Создай рыночный ордер на покупку<br>
                📋 Список доступных инструментов<br>
                ⏱️ История операций за сегодня
            </div>
        </div>
        """, unsafe_allow_html=True)

        api_key_present = bool(_env_value("OPENROUTER_API_KEY", "COMET_API_KEY", "LLM_API_KEY"))
        
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <div style="color: white; font-weight: 600; margin-bottom: 0.5rem;">🔗 Статус подключения</div>
        </div>
        """, unsafe_allow_html=True)
        
        if api_key_present:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10B981, #059669); color: white; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                ✅ <strong>OpenRouter API подключен</strong><br>
                <small>Готов к работе с ИИ</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #F59E0B, #D97706); color: white; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                ⚠️ <strong>Требуется настройка API</strong><br>
                <small>Укажите OPENROUTER_API_KEY</small>
            </div>
            """, unsafe_allow_html=True)
            
        # Connection status for Finam
        if st.session_state.finam_token:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10B981, #059669); color: white; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                🏦 <strong>Finam API настроен</strong><br>
                <small>Токен доступа установлен</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #6B7280, #4B5563); color: white; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                🔒 <strong>Finam API не настроен</strong><br>
                <small>Введите токен для доступа к торговле</small>
            </div>
            """, unsafe_allow_html=True)

    _render_history()

    # Chat input with enhanced styling
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <div style="text-align: center; color: #FFFFFF; font-size: 0.9rem; margin-bottom: 1rem; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);">
            💬 Введите ваш вопрос или команду для AI-ассистента
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    prompt = st.chat_input("Напишите ваш вопрос о торговле, портфеле или рынке...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #3B82F6, #1E3A8A); color: #FFFFFF; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);">
            {prompt}
        </div>
        """, unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="👾"):
        with st.spinner("🤔 Анализирую запрос и подготавливаю ответ..."):
            try:
                service = _get_service()
                response_text = service.process_request(prompt)
                tool_calls = call_logger.question_history(prompt)

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #3B82F6, #1E3A8A); color: #FFFFFF; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; border-left: 4px solid #10B981; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);">
                    {response_text}
                </div>
                """, unsafe_allow_html=True)

                if tool_calls:
                    with st.expander("🔧 Детали выполнения MCP инструментов", expanded=False):
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #FEF3C7, #FDE68A); color: #92400E; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; font-weight: 600;">
                            🛠️ Выполненные операции:
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for idx, call in enumerate(tool_calls, start=1):
                            st.markdown(f"""
                            <div style="background: #FFFFFF; color: #1F2937; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #10B981; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);">
                                <strong>#{idx} {call['tool']}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if call.get("params"):
                                st.json(call["params"])

                message_data: Dict[str, Any] = {"role": "assistant", "content": response_text}
                if tool_calls:
                    message_data["tool_calls"] = tool_calls
                st.session_state.messages.append(message_data)
            except Exception as exc:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #EF4444, #DC2626); color: #FFFFFF; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);">
                    ❌ <strong>Произошла ошибка:</strong><br>
                    <code style="background: rgba(255, 255, 255, 0.2); padding: 0.25rem 0.5rem; border-radius: 4px; color: #FFFFFF;">{exc}</code>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
