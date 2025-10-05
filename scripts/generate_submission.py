#!/usr/bin/env python3
"""Генерация submission.csv через новый многодоменный MCP-агент.

Скрипт запускает Finam MCP сервер, инициализирует специализированных
агентов (AUTH/ACCOUNTS/INSTRUMENTS/ORDERS/MARKET_DATA), прогоняет каждый
вопрос из test.csv, фиксирует последний вызов MCP инструмента и
преобразует его в строку HTTP запроса (формат train.csv).

По умолчанию используются переменные окружения (поддерживаются алиасы):
  - COMET_API_KEY / OPENROUTER_API_KEY / LLM_API_KEY – API ключ
  - COMET_MODEL_ID / OPENROUTER_MODEL / LLM_MODEL_ID – ID модели (default: qwen2.5-32b-instruct)
  - COMET_BASE_URL / OPENROUTER_BASE / LLM_BASE_URL – Базовый URL (default: https://api.cometapi.com/v1)
  - DEFAULT_ACCOUNT_ID    – подставляется, если агент не указал account_id
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from textwrap import dedent

import click
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.app.interfaces.call_logger import call_logger
from src.app.interfaces.mcp_agent import MCPOutputParser

try:  # pragma: no cover - tqdm опционален
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):  # type: ignore[override]
        return iterable

# ---------------------------------------------------------------------------
# Пути и окружение
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SERVER_SCRIPT = PROJECT_ROOT / "src" / "app" / "mcp" / "server.py"
DEFAULT_ACCOUNT_ID = os.getenv("DEFAULT_ACCOUNT_ID", "TRQD05:409933")

DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "SBER@MISX")
DEFAULT_UNDERLYING_SYMBOL = os.getenv("DEFAULT_UNDERLYING_SYMBOL", DEFAULT_SYMBOL)
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "D")
DEFAULT_ORDER_ID = os.getenv("DEFAULT_ORDER_ID", "ORDER123")
DEFAULT_ORDER_QUANTITY = os.getenv("DEFAULT_ORDER_QUANTITY", "1")
DEFAULT_ORDER_SIDE = os.getenv("DEFAULT_ORDER_SIDE", "BUY")
DEFAULT_ORDER_TYPE = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")
DEFAULT_ORDER_TIME_IN_FORCE = os.getenv("DEFAULT_ORDER_TIME_IN_FORCE", "DAY")
DEFAULT_AUTH_SECRET = os.getenv("DEFAULT_AUTH_SECRET", "demo-secret")
DEFAULT_SESSION_TOKEN = os.getenv("DEFAULT_SESSION_TOKEN", "demo-token")
DEFAULT_LIMIT_VALUE = os.getenv("DEFAULT_LIMIT_VALUE", "100")
DEFAULT_DEPTH_VALUE = os.getenv("DEFAULT_DEPTH_VALUE", "10")


def _env_value(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


COMET_BASE_URL = _env_value(
    "OPENROUTER_BASE",
    "LLM_BASE_URL",
    default="https://api.cometapi.com/v1",
)
COMET_MODEL_ID = _env_value(
    "OPENROUTER_MODEL",
    "LLM_MODEL_ID",
    "LLM_MODEL",
    default="qwen2.5-32b-instruct",
)
COMET_API_KEY = _env_value("OPENROUTER_API_KEY", "LLM_API_KEY")

# ---------------------------------------------------------------------------
# Логирование вызовов инструментов
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MCP → LangChain инструменты (адаптация входных схем)
# ---------------------------------------------------------------------------

_JSON_TO_TYPE: Dict[str, Type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

DEFAULT_FIELD_VALUES: Dict[str, Any] = {
    "account_id": DEFAULT_ACCOUNT_ID,
    "symbol": DEFAULT_SYMBOL,
    "underlying_symbol": DEFAULT_UNDERLYING_SYMBOL,
    "underlyingSymbol": DEFAULT_UNDERLYING_SYMBOL,
    "timeframe": DEFAULT_TIMEFRAME,
    "timeFrame": DEFAULT_TIMEFRAME,
    "order_id": DEFAULT_ORDER_ID,
    "orderId": DEFAULT_ORDER_ID,
    "quantity": DEFAULT_ORDER_QUANTITY,
    "side": DEFAULT_ORDER_SIDE,
    "type": DEFAULT_ORDER_TYPE,
    "time_in_force": DEFAULT_ORDER_TIME_IN_FORCE,
    "timeInForce": DEFAULT_ORDER_TIME_IN_FORCE,
    "secret": DEFAULT_AUTH_SECRET,
    "token": DEFAULT_SESSION_TOKEN,
    "limit": DEFAULT_LIMIT_VALUE,
    "depth": DEFAULT_DEPTH_VALUE,
}


def _jsonschema_to_args_schema(name: str, schema: Dict[str, Any] | None) -> Type[Any]:
    from pydantic import BaseModel, Field, create_model

    schema = schema or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: Dict[str, Tuple[Any, Any]] = {}

    for key, prop in props.items():
        json_type = prop.get("type", "string")
        py_type = _JSON_TO_TYPE.get(json_type, str)
        default = ... if key in required else None
        fields[key] = (py_type, Field(default, description=prop.get("description")))

    if not fields:
        return create_model(name)  # type: ignore[return-value]

    return create_model(name, **fields)  # type: ignore[return-value]


def _resp_to_text(response: Any) -> str:
    try:
        for content in getattr(response, "content", []) or []:
            if getattr(content, "type", None) == "text" and getattr(content, "text", None):
                return content.text
    except Exception:  # pragma: no cover - best effort
        pass
    return str(response)


def _tool_call_factory(
    session: ClientSession, tool_name: str, args_schema: Type[Any]
) -> Callable[..., Any]:
    async def _call(**kwargs: Any) -> str:
        params = dict(kwargs)
        fields = getattr(args_schema, "model_fields", {})
        if "account_id" in fields and "account_id" not in params:
            params["account_id"] = DEFAULT_ACCOUNT_ID
        for name in fields:
            if name in params:
                continue
            default_value = DEFAULT_FIELD_VALUES.get(name)
            if default_value is not None:
                params[name] = default_value
        try:
            call_logger.log_tool_call(tool_name, params)
        except Exception as log_exc:  # pragma: no cover - best effort
            print(f"⚠️  Не удалось записать вызов инструмента {tool_name}: {log_exc}")
        response = await session.call_tool(tool_name, params)
        if getattr(response, "isError", False):
            return f"Ошибка при вызове {tool_name}: {_resp_to_text(response)}"
        return _resp_to_text(response)

    return _call


async def create_tools_from_mcp(session: ClientSession) -> List[StructuredTool]:
    tools: List[StructuredTool] = []
    cursor: Optional[str] = None

    while True:
        listing = await session.list_tools(cursor=cursor)
        for tool in listing.tools:
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            ArgsSchema = _jsonschema_to_args_schema(f"{tool.name}Args", schema)
            coroutine = _tool_call_factory(session, tool.name, ArgsSchema)
            tools.append(
                StructuredTool(
                    name=tool.name,
                    description=tool.description or "MCP tool",
                    args_schema=ArgsSchema,
                    coroutine=coroutine,
                )
            )
        cursor = getattr(listing, "nextCursor", None)
        if not cursor:
            break

    return tools


# ---------------------------------------------------------------------------
# Доменные агенты
# ---------------------------------------------------------------------------


class AgentDomain(Enum):
    AUTH = "auth"
    ACCOUNTS = "accounts"
    INSTRUMENTS = "instruments"
    ORDERS = "orders"
    MARKET_DATA = "market_data"


TOOL_DOMAINS: Dict[str, AgentDomain] = {
    "Auth": AgentDomain.AUTH,
    "TokenDetails": AgentDomain.AUTH,
    "GetAccount": AgentDomain.ACCOUNTS,
    "Trades": AgentDomain.ACCOUNTS,
    "Transactions": AgentDomain.ACCOUNTS,
    "GetAssets": AgentDomain.INSTRUMENTS,
    "GetAsset": AgentDomain.INSTRUMENTS,
    "GetAssetParams": AgentDomain.INSTRUMENTS,
    "OptionsChain": AgentDomain.INSTRUMENTS,
    "Schedule": AgentDomain.INSTRUMENTS,
    "Clock": AgentDomain.INSTRUMENTS,
    "Exchanges": AgentDomain.INSTRUMENTS,
    "PlaceOrder": AgentDomain.ORDERS,
    "GetOrders": AgentDomain.ORDERS,
    "GetOrder": AgentDomain.ORDERS,
    "CancelOrder": AgentDomain.ORDERS,
    "Bars": AgentDomain.MARKET_DATA,
    "LastQuote": AgentDomain.MARKET_DATA,
    "LatestTrades": AgentDomain.MARKET_DATA,
    "OrderBook": AgentDomain.MARKET_DATA,
}


FALLBACK_TOOL_BY_DOMAIN: Dict[AgentDomain, str] = {
    AgentDomain.AUTH: "TokenDetails",
    AgentDomain.ACCOUNTS: "GetAccount",
    AgentDomain.INSTRUMENTS: "GetAsset",
    AgentDomain.ORDERS: "GetOrders",
    AgentDomain.MARKET_DATA: "LastQuote",
}


DOMAIN_DESCRIPTIONS = {
    AgentDomain.AUTH: "аутентификации и управления токенами",
    AgentDomain.ACCOUNTS: "работы со счетами, портфелями и балансами",
    AgentDomain.INSTRUMENTS: "поиска и анализа торговых инструментов",
    AgentDomain.ORDERS: "управления заявками (создание, отмена, мониторинг)",
    AgentDomain.MARKET_DATA: "получения рыночных данных (котировки, свечи, стаканы)",
}


def _domain_prompt(domain: AgentDomain, tools_desc: str, tool_names: str) -> str:
    return dedent(
            f"""
            Ты специализированный агент для {DOMAIN_DESCRIPTIONS[domain]}.

            Доступные инструменты:
            {tools_desc}

            Роснефть - ROSN@MISX
            Газпром - GAZP@MISX
            Газпром Нефть - SIBN@MISX
            Лукойл - LKOH@MISX
            Татнефть - TATN@MISX
            АЛРОСА - ALRS@MISX
            Сургутнефтегаз - SNGS@MISX
            РУСАЛ - RUAL@MISX
            Amazon - AMZN@XNGS
            ВТБ - VTBR@MISX
            Сбер / Сбербанк - SBERP@MISX, SBER@MISX
            Microsoft - MSFT@XNGS
            Аэрофлот - AFLT@MISX
            Магнит - MGNT@MISX
            Норникель - GMKN@MISX, GKZ5@RTSX (фьючерсы)
            Северсталь - CHZ5@RTSX (фьючерсы), CHMF@MISX
            ФосАгро - PHOR@MISX
            Юнипро - UPRO@MISX
            Распадская - RASP@MISX
            Полюс - PLZL@MISX
            X5 Retail Group
            ПИК - PIKK@MISX
            МТС - MTSS@MISX
            Новатэк - NVTK@MISX

            Используй JSON для вызова инструментов:
            ```
            {{{{
            "action": $TOOL_NAME,
            "action_input": $JSON_BLOB ("arg_name": "value")
            }}}}
            ```

            Valid "action" values: "Final Answer" или один из [{tool_names}]

            Формат работы:

            Question: входной вопрос
            Thought: анализ ситуации
            Action:
            $JSON_BLOB

            Observation: результат действия

            Action:
            ```
            {{{{
            "action": "Final Answer",
            "action_input": "Финальный ответ пользователю"
            }}}}
            ```

            История диалога:
            {{chat_history}}

            ВАЖНО:
            - Отвечай ТОЛЬКО на вопросы в твоей области ({DOMAIN_DESCRIPTIONS[domain]})
            - Всегда используй инструменты для получения актуальных данных
            - ВСЕГДА ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
            - Форматируй ответы понятно и структурированно
            - Если данных недостаточно, не переспрашивай пользователя: используй значения по умолчанию, НО ТОЛЬКО ЕСЛИ ОНИ ПОДХОДЯТ
              (symbol: {DEFAULT_SYMBOL}, timeframe: {DEFAULT_TIMEFRAME}, order_id: {DEFAULT_ORDER_ID},
              quantity: {DEFAULT_ORDER_QUANTITY}, side: {DEFAULT_ORDER_SIDE}, type: {DEFAULT_ORDER_TYPE},
              time_in_force: {DEFAULT_ORDER_TIME_IN_FORCE}) и иначе придумай сам из конетекста и сразу выполняй вызов подходящего инструмента
            - В случае любой ошибки сразу выдай Final Answer и сообщи пользователю об ошибке. Ни за что не повторяй один и тот же запрос повторно.
            - Если не указан ID аккаунта, используй значение по умолчанию: {DEFAULT_ACCOUNT_ID}
            - ЕСЛИ ТЕБЕ НЕ ХВАТАЕТ ИНФОРМАЦИИ — используй разумные значения по умолчанию и делай лучший доступный запрос.

            Thought:
            """
        ).strip()


@dataclass
class SpecializedAgent:
    domain: AgentDomain
    tools: List[StructuredTool]
    llm: ChatOpenAI

    def __post_init__(self) -> None:
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=3,
        )
        tool_names = ", ".join(tool.name for tool in self.tools)
        tools_desc = "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools)
        system_prompt = _domain_prompt(self.domain, tools_desc, tool_names)

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            handle_parsing_errors=True,
            verbose=False,
            max_iterations=5,
            agent_kwargs={
                "memory_prompts": ["chat_history"],
                "input_variables": ["input", "agent_scratchpad", "chat_history"],
            },
        )

        prompt = getattr(self.agent.agent.llm_chain, "prompt", None)
        if prompt is not None and getattr(prompt, "messages", None):
            first_message = prompt.messages[0]
            if hasattr(first_message, "prompt") and hasattr(first_message.prompt, "template"):
                first_message.prompt.template = system_prompt
            elif hasattr(first_message, "content"):
                first_message.content = system_prompt
            input_variables = getattr(prompt, "input_variables", None)
            if isinstance(input_variables, list) and "chat_history" not in input_variables:
                input_variables.append("chat_history")

        parser = getattr(self.agent.agent, "output_parser", None)
        if parser is not None and not isinstance(parser, MCPOutputParser):
            self.agent.agent.output_parser = MCPOutputParser(parser)

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        task_input = task
        if context and context.get("global_history"):
            task_input = f"Контекст:\n{context['global_history']}\n\nЗапрос: {task}"
        call_logger.clear_question_history(task)
        token = call_logger.set_current_question(task)
        try:
            result = await self.agent.ainvoke({"input": task_input})
        except Exception as exc:  # pylint: disable=broad-except
            print("⚠️  SpecializedAgent: ошибка выполнения агента в скрипте submission.")
            print("   ↳ домен:", self.domain.value)
            print("   ↳ входной запрос:\n", task_input)
            print("   ↳ тип исключения:", repr(exc))
            print("   ↳ traceback:\n", traceback.format_exc())
            history = call_logger.question_history(task)
            if history:
                print("  ↳ вызовы инструментов:")
                print(json.dumps(history, ensure_ascii=False, indent=2))
            else:
                print("   ↳ инструменты не вызывались")
            raise
        finally:
            if not call_logger.question_history(task):
                self._record_fallback_call()
            call_logger.reset_current_question(token)
        return result.get("output", str(result))

    def _record_fallback_call(self) -> None:
        tool = self._fallback_tool()
        if not tool:
            return
        params = self._default_params_for_tool(tool)
        call_logger.log_tool_call(tool.name, params)
        print(
            f"⚠️  Fallback: инструмент {tool.name} вызван с параметрами по умолчанию для домена '{self.domain.value}'."
        )

    def _fallback_tool(self) -> Optional[StructuredTool]:
        preferred = FALLBACK_TOOL_BY_DOMAIN.get(self.domain)
        if preferred:
            for tool in self.tools:
                if tool.name == preferred:
                    return tool
        return self.tools[0] if self.tools else None

    @staticmethod
    def _default_params_for_tool(tool: StructuredTool) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        fields = getattr(tool, "args_schema", None)
        model_fields = getattr(fields, "model_fields", {}) if fields is not None else {}

        if "account_id" in model_fields:
            params["account_id"] = DEFAULT_ACCOUNT_ID

        for name in model_fields:
            if name in params:
                continue
            default_value = DEFAULT_FIELD_VALUES.get(name)
            if default_value is not None:
                params[name] = default_value

        return params



class OrchestratorAgent:
    DOMAIN_MAP = {
        "AUTH": AgentDomain.AUTH,
        "ACCOUNTS": AgentDomain.ACCOUNTS,
        "INSTRUMENTS": AgentDomain.INSTRUMENTS,
        "ORDERS": AgentDomain.ORDERS,
        "MARKET_DATA": AgentDomain.MARKET_DATA,
    }

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.specialized_agents: Dict[AgentDomain, SpecializedAgent] = {}
        self.global_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10,
        )

    def add_agent(self, agent: SpecializedAgent) -> None:
        self.specialized_agents[agent.domain] = agent

    def _history_snapshot(self, max_messages: int = 6, max_length: int = 200) -> str:
        history = self.global_memory.load_memory_variables({}).get("chat_history") or []
        if not history:
            return "Нет предыдущих сообщений"
        lines: List[str] = []
        for msg in history[-max_messages:]:
            role = "Пользователь" if getattr(msg, "type", "human") == "human" else "Ассистент"
            content = (msg.content or "")[:max_length]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def route_request(self, user_input: str) -> AgentDomain:
        routing_prompt = dedent(
            f"""
            Ты агент-маршрутизатор в системе управления торговым счетом Finam.

            Доступные специализированные агенты:
            1. AUTH - аутентификация и токены (получение JWT, проверка токенов)
            2. ACCOUNTS - счета и портфели (баланс, позиции, транзакции, история сделок)
            3. INSTRUMENTS - торговые инструменты (поиск акций, параметры инструментов, расписание торгов, опционные цепочки)
            4. ORDERS - заявки (создание, отмена, просмотр активных заявок)
            5. MARKET_DATA - рыночные данные (котировки, свечи, стакан, последние сделки)

            История диалога:
            {self._history_snapshot()}

            Запрос пользователя: {user_input}

            Ответь ТОЛЬКО одним словом из списка: AUTH, ACCOUNTS, INSTRUMENTS, ORDERS, MARKET_DATA.
            """
        ).strip()

        response = await self.llm.ainvoke(routing_prompt)
        domain_key = str(getattr(response, "content", "")).strip().upper()
        return self.DOMAIN_MAP.get(domain_key, AgentDomain.ACCOUNTS)

    async def process_request(self, user_input: str) -> str:
        self.global_memory.chat_memory.add_user_message(user_input)
        domain = await self.route_request(user_input)
        agent = self.specialized_agents.get(domain)
        if not agent:
            message = f"Агент для домена {domain.value} не найден"
            self.global_memory.chat_memory.add_ai_message(message)
            return message
        context = {"global_history": self._history_snapshot()}
        result = await agent.execute(user_input, context)
        self.global_memory.chat_memory.add_ai_message(result)
        return result


def group_tools_by_domain(tools: Iterable[StructuredTool]) -> Dict[AgentDomain, List[StructuredTool]]:
    grouped: Dict[AgentDomain, List[StructuredTool]] = {domain: [] for domain in AgentDomain}
    for tool in tools:
        domain = TOOL_DOMAINS.get(tool.name)
        if domain:
            grouped[domain].append(tool)
    return grouped


# ---------------------------------------------------------------------------
# Преобразование вызова инструмента в HTTP запрос
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _placeholder(name: str) -> str:
    return f"{{{name}}}"


def _norm_symbol(value: Any) -> str:
    symbol = _stringify(value)
    return symbol.upper() if symbol else _placeholder("symbol")


def _norm_account(value: Any) -> str:
    account = _stringify(value)
    return account if account else _placeholder("account_id")


def _norm_order(value: Any) -> str:
    order = _stringify(value)
    return order if order else _placeholder("order_id")


def _norm_timeframe(value: Any) -> Optional[str]:
    timeframe = _stringify(value)
    return timeframe.upper() if timeframe else None


def _build_query(base: str, pairs: Iterable[Tuple[str, Any]]) -> str:
    parts = []
    for key, raw in pairs:
        value = _stringify(raw)
        if value is None:
            continue
        parts.append(f"{key}={value}")
    if not parts:
        return base
    query = "&".join(parts)
    return f"{base}?{query}" if "?" not in base else f"{base}&{query}"


def _extract_param(params: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in params:
            return params[key]
    return None


def _build_auth(_: Dict[str, Any]) -> Tuple[str, str]:
    return "POST", "/v1/sessions"


def _build_token_details(_: Dict[str, Any]) -> Tuple[str, str]:
    return "POST", "/v1/sessions/details"


def _build_get_account(params: Dict[str, Any]) -> Tuple[str, str]:
    account_id = _norm_account(_extract_param(params, "account_id", "accountId"))
    return "GET", f"/v1/accounts/{account_id}"


def _build_trades(params: Dict[str, Any]) -> Tuple[str, str]:
    account_id = _norm_account(_extract_param(params, "account_id", "accountId"))
    base = f"/v1/accounts/{account_id}/trades"
    return "GET", _build_query(
        base,
        [
            ("interval.start_time", _extract_param(params, "interval_start", "interval.start_time")),
            ("interval.end_time", _extract_param(params, "interval_end", "interval.end_time")),
            ("limit", params.get("limit")),
        ],
    )


def _build_transactions(params: Dict[str, Any]) -> Tuple[str, str]:
    account_id = _norm_account(_extract_param(params, "account_id", "accountId"))
    base = f"/v1/accounts/{account_id}/transactions"
    return "GET", _build_query(
        base,
        [
            ("interval.start_time", _extract_param(params, "interval_start", "interval.start_time")),
            ("interval.end_time", _extract_param(params, "interval_end", "interval.end_time")),
            ("limit", params.get("limit")),
        ],
    )


def _build_get_assets(_: Dict[str, Any]) -> Tuple[str, str]:
    return "GET", "/v1/assets"


def _build_get_asset(params: Dict[str, Any]) -> Tuple[str, str]:
    symbol = _norm_symbol(_extract_param(params, "symbol"))
    base = f"/v1/assets/{symbol}"
    account_id = _extract_param(params, "account_id", "accountId")
    return "GET", _build_query(base, [("account_id", account_id)])


def _build_get_asset_params(params: Dict[str, Any]) -> Tuple[str, str]:
    symbol = _norm_symbol(_extract_param(params, "symbol"))
    base = f"/v1/assets/{symbol}/params"
    account_id = _extract_param(params, "account_id", "accountId")
    return "GET", _build_query(base, [("account_id", account_id)])


def _build_options_chain(params: Dict[str, Any]) -> Tuple[str, str]:
    symbol = _norm_symbol(_extract_param(params, "underlying_symbol", "symbol"))
    return "GET", f"/v1/assets/{symbol}/options"


def _build_schedule(params: Dict[str, Any]) -> Tuple[str, str]:
    symbol = _norm_symbol(_extract_param(params, "symbol"))
    return "GET", f"/v1/assets/{symbol}/schedule"


def _build_clock(_: Dict[str, Any]) -> Tuple[str, str]:
    return "GET", "/v1/assets/clock"


def _build_exchanges(_: Dict[str, Any]) -> Tuple[str, str]:
    return "GET", "/v1/exchanges"


def _build_get_orders(params: Dict[str, Any]) -> Tuple[str, str]:
    account_id = _norm_account(_extract_param(params, "account_id", "accountId"))
    return "GET", f"/v1/accounts/{account_id}/orders"


def _build_get_order(params: Dict[str, Any]) -> Tuple[str, str]:
    account_id = _norm_account(_extract_param(params, "account_id", "accountId"))
    order_id = _norm_order(_extract_param(params, "order_id", "orderId"))
    return "GET", f"/v1/accounts/{account_id}/orders/{order_id}"


def _build_cancel_order(params: Dict[str, Any]) -> Tuple[str, str]:
    account_id = _norm_account(_extract_param(params, "account_id", "accountId"))
    order_id = _norm_order(_extract_param(params, "order_id", "orderId"))
    return "DELETE", f"/v1/accounts/{account_id}/orders/{order_id}"


def _build_place_order(params: Dict[str, Any]) -> Tuple[str, str]:
    account_id = _norm_account(_extract_param(params, "account_id", "accountId"))
    return "POST", f"/v1/accounts/{account_id}/orders"


def _build_last_quote(params: Dict[str, Any]) -> Tuple[str, str]:
    symbol = _norm_symbol(_extract_param(params, "symbol"))
    return "GET", f"/v1/instruments/{symbol}/quotes/latest"


def _build_orderbook(params: Dict[str, Any]) -> Tuple[str, str]:
    symbol = _norm_symbol(_extract_param(params, "symbol"))
    base = f"/v1/instruments/{symbol}/orderbook"
    return "GET", _build_query(base, [("depth", params.get("depth"))])


def _build_latest_trades(params: Dict[str, Any]) -> Tuple[str, str]:
    symbol = _norm_symbol(_extract_param(params, "symbol"))
    return "GET", f"/v1/instruments/{symbol}/trades/latest"


def _build_bars(params: Dict[str, Any]) -> Tuple[str, str]:
    symbol = _norm_symbol(_extract_param(params, "symbol"))
    base = f"/v1/instruments/{symbol}/bars"
    return "GET", _build_query(
        base,
        [
            ("timeframe", _norm_timeframe(_extract_param(params, "timeframe"))),
            ("interval.start_time", _extract_param(params, "interval_start", "interval.start_time")),
            ("interval.end_time", _extract_param(params, "interval_end", "interval.end_time")),
            ("limit", params.get("limit")),
        ],
    )


TOOL_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Tuple[str, str]]] = {
    "Auth": _build_auth,
    "TokenDetails": _build_token_details,
    "GetAccount": _build_get_account,
    "Trades": _build_trades,
    "Transactions": _build_transactions,
    "GetAssets": _build_get_assets,
    "GetAsset": _build_get_asset,
    "GetAssetParams": _build_get_asset_params,
    "OptionsChain": _build_options_chain,
    "Schedule": _build_schedule,
    "Clock": _build_clock,
    "Exchanges": _build_exchanges,
    "GetOrders": _build_get_orders,
    "GetOrder": _build_get_order,
    "CancelOrder": _build_cancel_order,
    "PlaceOrder": _build_place_order,
    "LastQuote": _build_last_quote,
    "OrderBook": _build_orderbook,
    "LatestTrades": _build_latest_trades,
    "Bars": _build_bars,
}

DEFAULT_METHOD = "GET"
DEFAULT_PATH = "/v1/assets/clock"


def _extract_request(question: str) -> Tuple[str, str]:
    history = call_logger.question_history(question)
    last_call = history[-1] if history else None
    if not last_call:
        return DEFAULT_METHOD, DEFAULT_PATH
    tool = last_call.get("tool")
    params = last_call.get("params", {})
    builder = TOOL_BUILDERS.get(tool)
    if not builder:
        return DEFAULT_METHOD, DEFAULT_PATH
    try:
        method, path = builder(params)
    except Exception:  # pragma: no cover - дефолт на неожиданные данные
        return DEFAULT_METHOD, DEFAULT_PATH
    return method or DEFAULT_METHOD, path or DEFAULT_PATH


def _format_request(method: str, path: str) -> str:
    method_clean = (method or DEFAULT_METHOD).strip().upper()
    path_clean = path or DEFAULT_PATH
    return f"{method_clean} {path_clean}"


# ---------------------------------------------------------------------------
# Основная логика генерации
# ---------------------------------------------------------------------------


def _load_questions(test_file: Path) -> List[Dict[str, str]]:
    questions: List[Dict[str, str]] = []
    with test_file.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            questions.append({"uid": row["uid"], "question": row["question"]})
    return questions


def _write_submission(output_file: Path, rows: List[Dict[str, str]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["uid", "type", "request"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


async def _generate_predictions(test_questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not COMET_API_KEY:
        raise RuntimeError(
            "Не найден API ключ (COMET_API_KEY / OPENROUTER_API_KEY / LLM_API_KEY)"
        )
    if not SERVER_SCRIPT.exists():
        raise FileNotFoundError(f"Не найден MCP сервер: {SERVER_SCRIPT}")

    llm = ChatOpenAI(
        model=COMET_MODEL_ID,
        base_url=COMET_BASE_URL,
        api_key=COMET_API_KEY,
        temperature=0,
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        env=os.environ.copy(),
    )

    results: List[Dict[str, str]] = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await create_tools_from_mcp(session)
            if not tools:
                raise RuntimeError("Не удалось загрузить MCP инструменты")

            tools_by_domain = group_tools_by_domain(tools)
            orchestrator = OrchestratorAgent(llm)

            for domain, domain_tools in tools_by_domain.items():
                if domain_tools:
                    agent = SpecializedAgent(domain, domain_tools, llm)
                    orchestrator.add_agent(agent)

            for item in tqdm(test_questions, desc="Обработка"):
                uid = item["uid"]
                question = item["question"].strip()
                if not question:
                    method, path = DEFAULT_METHOD, DEFAULT_PATH
                else:
                    call_logger.clear_question_history(question)
                    try:
                        await orchestrator.process_request(question)
                    except Exception as exc:  # pragma: no cover - агент уже логирует
                        click.echo(f"⚠️  Ошибка при обработке '{question[:60]}...': {exc}", err=True)
                    method, path = _extract_request(question)
                    call_logger.clear_question_history(question)

                results.append(
                    {
                        "uid": uid,
                        "type": method,
                        "request": _format_request(method, path),
                    }
                )

    return results


@click.command()
@click.option(
    "--test-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed/test.csv"),
    help="Путь к test.csv",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/processed/submission.csv"),
    help="Куда сохранить submission.csv",
)
def main(test_file: Path, output_file: Path) -> None:
    """Запуск генерации submission."""

    click.echo("🚀 Старт генерации submission через MCP агента...")
    questions = _load_questions(test_file)
    click.echo(f"📖 Загружено вопросов: {len(questions)}")

    if not questions:
        click.echo("⚠️  test.csv пуст, прекращаем", err=True)
        return

    rows = asyncio.run(_generate_predictions(questions))
    _write_submission(output_file, rows)
    click.echo(f"✅ Submission сохранён: {output_file}")


if __name__ == "__main__":
    main()
