from __future__ import annotations

import asyncio
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import StructuredTool, Tool
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

COMETAPI_BASE_URL = os.getenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen2.5-32b-instruct")
COMET_API_KEY = os.getenv("COMET_API_KEY", "sk-eda8aMPSz9nfgZwaVTAvkkLZtXMiiyLMLbna3GixHlfa7G2K")

# Default values for Finam API
DEFAULT_ACCOUNT_ID = os.getenv("DEFAULT_ACCOUNT_ID", "")
DEFAULT_FIELD_VALUES = {
    "account_id": DEFAULT_ACCOUNT_ID,
    "accountId": DEFAULT_ACCOUNT_ID,
}


class AgentDomain(Enum):
    """Домены специализированных агентов"""

    ACCOUNTS = "accounts"
    INSTRUMENTS = "instruments"
    ORDERS = "orders"
    MARKET_DATA = "market_data"
    AUTH = "auth"


TOOL_DOMAINS: Dict[str, AgentDomain] = {
    "Auth": AgentDomain.AUTH,
    "TokenDetails": AgentDomain.AUTH,
    "GetAccount": AgentDomain.ACCOUNTS,
    "Trades": AgentDomain.ACCOUNTS,
    "Transactions": AgentDomain.ACCOUNTS,
    # "Clock_ACCOUNTS": AgentDomain.ACCOUNTS,
    "GetAsset": AgentDomain.INSTRUMENTS,
    "GetAssetParams": AgentDomain.INSTRUMENTS,
    "OptionsChain": AgentDomain.INSTRUMENTS,
    "Schedule": AgentDomain.INSTRUMENTS,
    "Exchanges": AgentDomain.INSTRUMENTS,
    "PlaceOrder": AgentDomain.ORDERS,
    "GetOrders": AgentDomain.ORDERS,
    "GetOrder": AgentDomain.ORDERS,
    "CancelOrder": AgentDomain.ORDERS,
    "Bars": AgentDomain.MARKET_DATA,
    "LastQuote": AgentDomain.MARKET_DATA,
    "LatestTrades": AgentDomain.MARKET_DATA,
    "OrderBook": AgentDomain.MARKET_DATA,
    # "Clock_MARKET_DATA": AgentDomain.MARKET_DATA,
}


DOMAIN_DESCRIPTIONS: Dict[AgentDomain, str] = {
    AgentDomain.AUTH: "аутентификации и получения информации о токенах",
    AgentDomain.ACCOUNTS: "работы со счетами, портфелями и балансами",
    AgentDomain.INSTRUMENTS: "поиска и анализа торговых инструментов",
    AgentDomain.ORDERS: "управления заявками (создание, отмена, мониторинг)",
    AgentDomain.MARKET_DATA: "получения рыночных данных (котировки, свечи, стаканы)",
}


class SpecializedAgent:
    """Специализированный агент для конкретного домена"""

    def __init__(self, domain: AgentDomain, tools: List[Tool], llm: ChatOpenAI):
        self.domain = domain
        self.tools = tools
        self.llm = llm
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            k=3,
        )
        self.agent = self._create_agent()

    def _create_agent(self):
        """Создание агента с оптимизированной конфигурацией"""
        tool_names = ", ".join(t.name for t in self.tools)
        tools_desc = "\n".join(f"{t.name}: {t.description}" for t in self.tools)
        system_prompt = self._build_domain_prompt(tools_desc, tool_names)

        agent = initialize_agent(
            self.tools,
            self.llm,
            memory=self.memory,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=5,
            agent_kwargs={
                "memory_prompts": ["chat_history"],
                "input_variables": ["input", "agent_scratchpad", "chat_history"],
            },
        )

        agent.agent.llm_chain.prompt.messages[0].prompt.template = system_prompt

        if "chat_history" not in agent.agent.llm_chain.prompt.input_variables:
            agent.agent.llm_chain.prompt.input_variables.append("chat_history")

        return agent

    def _build_domain_prompt(self, tools_desc: str, tool_names: str) -> str:
        """Построение промпта для специализированного агента"""
        return f"""Ты специализированный агент для {DOMAIN_DESCRIPTIONS[self.domain]}.

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

МЕЧЕЛ - MTLR@MISX

YANDEX - YDEX@MISX

период за все время: 2014-01-01T00:00:00Z - сегодня 
для получения текущего времени используй 2025-10-04T11:20:44.421182013Z 

Доступные таймфреймы 

TIME_FRAME_M1\t1 минута. Глубина данных 7 дней.
TIME_FRAME_M5\t5 минут. Глубина данных 30 дней.
TIME_FRAME_M15\t15 минут. Глубина данных 30 дней.
TIME_FRAME_M30\t30 минут. Глубина данных 30 дней.
TIME_FRAME_H1\t1 час. Глубина данных 30 дней.
TIME_FRAME_H2\t2 часа. Глубина данных 30 дней.
TIME_FRAME_H4\t4 часа. Глубина данных 30 дней.
TIME_FRAME_H8\t8 часов. Глубина данных 30 дней.
TIME_FRAME_D\t1 День. Глубина данных 365 дней.
TIME_FRAME_W\tНеделя. Глубина данных 5 лет.
TIME_FRAME_MN\tМесяц. Глубина данных 5 лет.
TIME_FRAME_QR\tКвартал (3 месяца). Глубина данных 5 лет.


Используй JSON для вызова инструментов:
```
{{{{
"action": $TOOL_NAME,
"action_input": $JSON_BLOB ("arg_name": "value")
}}}}
```

Valid "action" values: "Final Answer" or one of [{tool_names}]

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
- Отвечай ТОЛЬКО на вопросы в твоей области ({DOMAIN_DESCRIPTIONS[self.domain]})
- Всегда используй инструменты для получения актуальных данных
- ВСЕГДА ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
- Форматируй ответы понятно и структурированно
- Если данных недостаточно, уточни у пользователя
- В случае получения любой ошибки выдавай Final Answer и сообщай пользователю об ошибке. ни за что не пробуй снова.ни за что не повторяй один и тот же запрос
- Если не указан айди аккаунта используй айди по умолчанию: TRQD05:409933 

Thought:
"""

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Выполнение задачи агентом"""
        task_input = task
        if context and "global_history" in context:
            task_input = f"Контекст из истории:\n{context['global_history']}\n\nТекущий запрос: {task}"

        result = await self.agent.ainvoke({"input": task_input})
        return result["output"]


class OrchestratorAgent:
    """Оркестратор для маршрутизации запросов между агентами"""

    DOMAIN_MAP: Dict[str, AgentDomain] = {
        "AUTH": AgentDomain.AUTH,
        "ACCOUNTS": AgentDomain.ACCOUNTS,
        "INSTRUMENTS": AgentDomain.INSTRUMENTS,
        "ORDERS": AgentDomain.ORDERS,
        "MARKET_DATA": AgentDomain.MARKET_DATA,
    }

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.specialized_agents: Dict[AgentDomain, SpecializedAgent] = {}
        self.global_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10,
        )

    def add_agent(self, agent: SpecializedAgent) -> None:
        """Добавление специализированного агента"""
        self.specialized_agents[agent.domain] = agent

    def _get_history(self, max_messages: int = 6, max_length: int = 200) -> str:
        """Получение истории диалога"""
        memory_vars = self.global_memory.load_memory_variables({})

        if not memory_vars.get("chat_history"):
            return "Нет предыдущих сообщений"

        history_text: List[str] = []
        for msg in memory_vars["chat_history"][-max_messages:]:
            role = "Пользователь" if msg.type == "human" else "Ассистент"
            content = msg.content[:max_length]
            history_text.append(f"{role}: {content}")

        return "\n".join(history_text)

    async def route_request(self, user_input: str) -> AgentDomain:
        """Маршрутизация запроса к соответствующему агенту"""
        routing_prompt = f"""Ты агент-маршрутизатор в системе управления торговым счетом Finam.

Доступные специализированные агенты:

1. AUTH - Аутентификация и авторизация
   • Получение JWT токенов доступа
   • Проверка и обновление токенов
   • Управление сессиями

2. ACCOUNTS - Управление счетами и портфелями
   • Получение информации о конкретном аккаунте (баланс, статус, equity)
   • Просмотр открытых позиций с деталями (количество, средняя цена, PnL)
   • Получение истории сделок за период (TradesRequest)
   • Просмотр списка транзакций (пополнения, выводы, комиссии, налоги)
   • Информация о типах портфелей: FORTS (срочный рынок), MC (Московская Биржа), MCT (американские рынки)
   • Доступные средства, маржинальные требования, нереализованная прибыль

3. INSTRUMENTS - Торговые инструменты и биржи
   • Поиск и получение списка доступных инструментов (акции, облигации, фьючерсы, опционы)
   • Детальная информация по инструменту (тикер, ISIN, тип, размер лота, шаг цены)
   • Получение торговых параметров (доступность для лонг/шорт, маржинальные требования)
   • Список доступных бирж и их MIC коды
   • Расписание торговых сессий для инструмента
   • Цепочки опционов для базовых активов

4. ORDERS - Управление заявками
   • Выставление новых заявок (рыночные, лимитные, стоп-заявки, мульти-лег)
   • Отмена активных заявок
   • Получение информации о конкретной заявке по ID
   • Просмотр списка всех заявок аккаунта
   • Поддержка типов: MARKET, LIMIT, STOP, STOP_LIMIT, MULTI_LEG
   • Настройка срока действия (DAY, GTC, IOC, FOK)
   • Отслеживание статусов (новая, частично исполнена, исполнена, отменена)

5. MARKET_DATA - Рыночные данные реального времени
   • Получение последней котировки (bid, ask, last price, объемы)
   • Исторические свечи (timeframes: M1, M5, M15, M30, H1, H2, H4, H8, D, W, MN, QR)
   • Стакан заявок (order book) с уровнями цен
   • Последние сделки по инструменту
   • Греки для опционов (delta, gamma, theta, vega, rho)
   • Дневная статистика (open, high, low, close, volume, turnover)

История диалога:
{self._get_history()}

Запрос пользователя: {user_input}

Проанализируй запрос и определи, какой агент должен его обработать.
Ответь ТОЛЬКО одним словом из списка: AUTH, ACCOUNTS, INSTRUMENTS, ORDERS, MARKET_DATA

Примеры маршрутизации:
- "покажи мой портфель" -> ACCOUNTS
- "какой у меня баланс" -> ACCOUNTS
- "покажи мои позиции" -> ACCOUNTS
- "история транзакций за июль" -> ACCOUNTS
- "последние сделки по счету" -> ACCOUNTS

- "купи 10 акций Сбербанка" -> ORDERS
- "выстави лимитную заявку на GAZP" -> ORDERS
- "отмени заявку 12345" -> ORDERS
- "покажи мои активные заявки" -> ORDERS
- "создай стоп-лосс" -> ORDERS

- "какая цена SBER" -> MARKET_DATA
- "покажи котировки Газпрома" -> MARKET_DATA
- "свечи YNDX за месяц" -> MARKET_DATA
- "стакан по LKOH" -> MARKET_DATA
- "последние сделки по ROSN" -> MARKET_DATA

- "найди акции Яндекса" -> INSTRUMENTS
- "можно ли купить TSLA" -> INSTRUMENTS
- "список доступных инструментов" -> INSTRUMENTS
- "расписание торгов SBER" -> INSTRUMENTS
- "опционы на Si" -> INSTRUMENTS
- "какие биржи доступны" -> INSTRUMENTS
- "параметры маржи для GAZP" -> INSTRUMENTS

- "авторизуйся" -> AUTH
- "получи токен" -> AUTH
- "обнови токен доступа" -> AUTH

Ответ:"""

        response = await self.llm.ainvoke(routing_prompt)
        domain_str = response.content.strip().upper()
        selected_domain = self.DOMAIN_MAP.get(domain_str, AgentDomain.ACCOUNTS)

        print(f"\n🎯 Оркестратор направил запрос агенту: {selected_domain.value}")
        return selected_domain

    async def process_request(self, user_input: str) -> str:
        """Обработка пользовательского запроса"""
        try:
            self.global_memory.chat_memory.add_user_message(user_input)
            target_domain = await self.route_request(user_input)

            agent = self.specialized_agents.get(target_domain)

            if not agent:
                error_msg = f"Агент для домена {target_domain.value} не найден"
                self.global_memory.chat_memory.add_ai_message(error_msg)
                return error_msg

            context = {"global_history": self._get_history()}
            result = await agent.execute(user_input, context)
            self.global_memory.chat_memory.add_ai_message(result)

            return result

        except Exception as exc:  # pragma: no cover - defensive logging
            error_msg = f"Произошла ошибка при обработке запроса: {str(exc)}"
            print(f"❌ Ошибка: {error_msg}")
            self.global_memory.chat_memory.add_ai_message(error_msg)
            return error_msg


def create_tool_wrapper(session: ClientSession, tool_name: str):
    """Фабрика для создания wrapper-функции инструмента"""

    async def _call_func(*args, **kwargs):
        try:
            params: Dict[str, Any] = {}

            if args and isinstance(args[0], dict):
                params = args[0]
            elif args and isinstance(args[0], str):
                try:
                    params = json.loads(args[0])
                except json.JSONDecodeError:
                    params = {"symbol": args[0]}
            elif kwargs:
                params = kwargs
            elif args:
                params = {"input": str(args[0])}

            print(f"🔧 Tool call: {tool_name}, params: {params}")

            response = await session.call_tool(tool_name, params)

            if hasattr(response, "isError") and response.isError:
                error_content = ""
                if hasattr(response, "content") and response.content:
                    for content_item in response.content:
                        if hasattr(content_item, "text"):
                            error_content = content_item.text
                            break
                return f"Ошибка при вызове {tool_name}: {error_content}"

            return str(response)
        except Exception as exc:  # pragma: no cover - defensive logging
            error_msg = f"Ошибка при вызове инструмента {tool_name}: {str(exc)}"
            print(f"❌ {error_msg}")
            return error_msg

    return _call_func


_JSON_TO_PY: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def jsonschema_to_args_schema(name: str, schema: Dict[str, Any] | None) -> Type[BaseModel]:
    schema = schema or {}
    props: Dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: Dict[str, tuple[type, Field]] = {}

    for key, prop in props.items():
        json_type = prop.get("type", "string")
        py_type = _JSON_TO_PY.get(json_type, str)
        default = ... if key in required else None
        fields[key] = (py_type, Field(default, description=prop.get("description")))

    if not fields:
        fields["input"] = (str, Field(..., description="Free-form input"))

    return create_model(name, **fields)  # type: ignore


def _mcp_response_to_text(resp: Any) -> str:
    try:
        for content in getattr(resp, "content", []) or []:
            if getattr(content, "type", None) == "text" and getattr(content, "text", None):
                return content.text
    except Exception:
        pass
    return str(resp)


def _structured_call_factory(session: ClientSession, tool_name: str):
    async def _call(**kwargs):
        print(f"🔧 Tool call: {tool_name}, params: {kwargs}")
        resp = await session.call_tool(tool_name, kwargs)
        return _mcp_response_to_text(resp)

    return _call


async def create_tools_from_mcp(session: ClientSession) -> List[StructuredTool]:
    tools_result = await session.list_tools()
    structured_tools: List[StructuredTool] = []

    for tool in tools_result.tools:
        tool_name = tool.name
        input_schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        args_schema = jsonschema_to_args_schema(f"{tool_name}Args", input_schema)

        call = _structured_call_factory(session, tool_name)
        structured_tools.append(
            StructuredTool(
                name=tool_name,
                description=tool.description or "MCP tool",
                args_schema=args_schema,
                coroutine=call,
            )
        )
        print(f"✅ Зарегистрирован StructuredTool: {tool_name}")

    return structured_tools


def group_tools_by_domain(tools: List[Tool]) -> Dict[AgentDomain, List[Tool]]:
    """Группировка инструментов по доменам"""
    grouped: Dict[AgentDomain, List[Tool]] = {domain: [] for domain in AgentDomain}

    for tool in tools:
        domain = TOOL_DOMAINS.get(tool.name)
        if domain:
            grouped[domain].append(tool)

    return grouped


async def run_test_queries(orchestrator: OrchestratorAgent, queries: List[str]) -> None:
    """Запуск тестовых запросов"""
    for idx, query in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 Запрос {idx}: {query}")
        print("=" * 70)

        try:
            result = await orchestrator.process_request(query)
            print(f"\n💬 Ответ: {result}")
        except Exception as exc:  # pragma: no cover - debug helper
            print(f"\n❌ Ошибка при обработке запроса: {exc}")

        print("-" * 70)
        await asyncio.sleep(1)


async def run_interactive_mode(orchestrator: OrchestratorAgent) -> None:
    """Интерактивный режим общения"""
    print("\n" + "=" * 70)
    print("🎮 Интерактивный режим (введите 'exit' для выхода)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
            if user_input.lower() in {"exit", "quit", "выход"}:
                print("👋 До свидания!")
                break

            if not user_input:
                continue

            result = await orchestrator.process_request(user_input)
            print(f"\n🤖 Ассистент: {result}")

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as exc:  # pragma: no cover - interactive safety
            print(f"\n❌ Ошибка: {exc}")


SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "mcp" / "server.py"
PYTHON_EXEC = sys.executable or "python"


def build_llm() -> ChatOpenAI:
    """Создает и возвращает настроенную модель LLM"""
    return ChatOpenAI(
        model=MODEL_ID,
        base_url=COMETAPI_BASE_URL,
        api_key=COMET_API_KEY,
        temperature=0,
    )


async def main() -> None:
    """Главная функция запуска системы"""
    llm = build_llm()

    server_params = StdioServerParameters(
        command=PYTHON_EXEC,
        args=[str(SERVER_SCRIPT)],
        env=None,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                structured_tools = await create_tools_from_mcp(session)

                if not structured_tools:
                    print("❌ Не удалось загрузить инструменты из MCP сервера")
                    return

                default_secret = os.getenv("FINAM_AUTH_SECRET") or os.getenv("FINAM_ACCESS_TOKEN")
                if default_secret:
                    try:
                        await session.call_tool("Auth", {"secret": default_secret})
                        print("🔐 Выполнена автоматическая авторизация MCP")
                    except Exception as auth_exc:  # pragma: no cover - auth helper
                        print(f"⚠️ Не удалось выполнить автоматическую авторизацию: {auth_exc}")

                tools_by_domain = group_tools_by_domain(structured_tools)
                orchestrator = OrchestratorAgent(llm)

                for domain, domain_tools in tools_by_domain.items():
                    if domain_tools:
                        agent = SpecializedAgent(domain, domain_tools, llm)
                        orchestrator.add_agent(agent)
                        print(f"✅ Создан агент {domain.value} с {len(domain_tools)} инструментами")

                print("\n" + "=" * 70)
                print("🚀 Мультиагентная система готова к работе!")
                print("=" * 70)

                await run_interactive_mode(orchestrator)

    except Exception as exc:  # pragma: no cover - startup errors
        print(f"\n❌ Критическая ошибка: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


def main_cli() -> None:
    """Poetry entry point wrapper."""
    asyncio.run(main())
