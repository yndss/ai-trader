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
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")
MODEL_ID = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_API_KEY = os.getenv("COMET_API_KEY", "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")
 


class AgentDomain(Enum):
    """Домены специализированных агентов"""
    ACCOUNTS = "accounts"
    INSTRUMENTS = "instruments"
    ORDERS = "orders"
    MARKET_DATA = "market_data"
    AUTH = "auth"


TOOL_DOMAINS = {
    "Auth": AgentDomain.AUTH,
    "TokenDetails": AgentDomain.AUTH,    
    "GetAccount": AgentDomain.ACCOUNTS,
    "Trades": AgentDomain.ACCOUNTS,
    "Transactions": AgentDomain.ACCOUNTS, 
    "Clock_ACCOUNTS": AgentDomain.ACCOUNTS,

    "Assets": AgentDomain.INSTRUMENTS,
    "Clock": AgentDomain.INSTRUMENTS,
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
    "Clock_MARKET_DATA": AgentDomain.MARKET_DATA,

}

DOMAIN_DESCRIPTIONS = {
    AgentDomain.AUTH: "аутентификации и получения информации о токенах",
    AgentDomain.ACCOUNTS: "работы со счетами, портфелями и балансами",
    AgentDomain.INSTRUMENTS: "поиска и анализа торговых инструментов",
    AgentDomain.ORDERS: "управления заявками (создание, отмена, мониторинг)",
    AgentDomain.MARKET_DATA: "получения рыночных данных (котировки, свечи, стаканы)"
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
            k=3
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
                "input_variables": ["input", "agent_scratchpad", "chat_history"]
            }
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

Яндекс - YDEX@MISX
Ozon - OZON@MISX 
VK - VKCO@MISX 
Московская биржа - MOEX@MISX 
АФК «Система» - AFKS@MISX 
МКБ - CBOM@MISX 
Русагро - RAGR@MISX
ФСК ЕЭС - FEES@MISX 
НЛМК - NLMK@MISX 
Транснефть - TRNFP@MISX 
Полиметалл - PLZL@MISX 
TCS Group (Тинькофф) - T@MISX
Интер РАО - IRAO@MISX 
X5 Retail Group - X5@MISX 
Apple - AAPL@XNGS 
Tesla - TSLA@XNGS 
HeadHunter - HEAD@MISX 
Amazon - AMZN@XNGS

фьючерсы 
Фьючерс индекс РТС - RIZ5@RTSX
Фьючерс доллар-рубль - SiZ5@RTSX
Фьючерс Brent - BRZ5@RTSX
Фьючерс природный газ - NGZ5@RTSX
Фьючерс евро-рубль - EuZ5@RTSX
Юань/Рубль (TOM) - CNYRUB_TOM@MISX
Золото/Рубль (TOM) - GLDRUB_TOM@MISX

период за все время: 2014-01-01T00:00:00Z - сегодня 

Доступные таймфреймы 

TIME_FRAME_M1	1 минута. Глубина данных 7 дней.
TIME_FRAME_M5	5 минут. Глубина данных 30 дней.
TIME_FRAME_M15	15 минут. Глубина данных 30 дней.
TIME_FRAME_M30	30 минут. Глубина данных 30 дней.
TIME_FRAME_H1	1 час. Глубина данных 30 дней.
TIME_FRAME_H2	2 часа. Глубина данных 30 дней.
TIME_FRAME_H4	4 часа. Глубина данных 30 дней.
TIME_FRAME_H8	8 часов. Глубина данных 30 дней.
TIME_FRAME_D	1 День. Глубина данных 365 дней.
TIME_FRAME_W	Неделя. Глубина данных 5 лет.
TIME_FRAME_MN	Месяц. Глубина данных 5 лет.
TIME_FRAME_QR	Квартал (3 месяца). Глубина данных 5 лет.


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
    
    DOMAIN_MAP = {
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
            k=10
        )
    
    def add_agent(self, agent: SpecializedAgent) -> None:
        """Добавление специализированного агента"""
        self.specialized_agents[agent.domain] = agent
    
    def _get_history(self, max_messages: int = 6, max_length: int = 200) -> str:
        """Получение истории диалога"""
        memory_vars = self.global_memory.load_memory_variables({})
        
        if not memory_vars.get("chat_history"):
            return "Нет предыдущих сообщений"
        
        history_text = []
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
   • Просмотр времени

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
    
    async def process_request(self, user_input: str, query_id = "") -> str:
        """Обработка пользовательского запроса"""
        try:
            token = current_query_id.set(query_id)
            
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
            finally:
                current_query_id.reset(token)
            
        except Exception as e:
            error_msg = f"Произошла ошибка при обработке запроса: {str(e)}"
            print(f"❌ Ошибка: {error_msg}")
            self.global_memory.chat_memory.add_ai_message(error_msg)
            return error_msg

from pydantic import create_model, Field
from typing import Any, Dict, Tuple, Type

import contextvars
from typing import Optional

current_query_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'current_query_id', 
    default=None
)


def create_tool_wrapper(session: ClientSession, tool_name: str):
    """Фабрика для создания wrapper-функции инструмента"""
    async def _call_func(*args, **kwargs):
        try:
            
            params = {}
            
            if args and isinstance(args[0], dict):
                params = args[0]
            elif args and isinstance(args[0], str):
                import json
                try:
                    params = json.loads(args[0])
                except json.JSONDecodeError:
                    params = {"symbol": args[0]}
            elif kwargs:
                params = kwargs
            elif args:
                params = {"input": str(args[0])}
            
            response = await session.call_tool(tool_name, params)
            
  
            
            if hasattr(response, 'isError') and response.isError:
                error_content = ""
                if hasattr(response, 'content') and response.content:
                    for content_item in response.content:
                        if hasattr(content_item, 'text'):
                            error_content = content_item.text
                            break
                return f"Ошибка при вызове {tool_name}: {error_content}"
            
            return str(response)
        except Exception as e:
            query_id = current_query_id.get()
            error_msg = f"Ошибка при вызове инструмента {tool_name}: {str(e)}"
            print(f"❌ [Query: {query_id}] {error_msg}")
            return error_msg
    
    return _call_func
    

from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field, create_model
from langchain.tools import StructuredTool
from functools import partial

_JSON_TO_PY = {
    "string": str, "integer": int, "number": float, "boolean": bool,
    "object": dict, "array": list,
}

def jsonschema_to_args_schema(name: str, schema: Dict[str, Any] | None) -> Type[BaseModel]:
    schema = schema or {}
    props: Dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: Dict[str, tuple[type, Field]] = {}

    for key, prop in props.items():
        jt = prop.get("type", "string")
        py_t = _JSON_TO_PY.get(jt, str)
        default = ... if key in required else None
        fields[key] = (py_t, Field(default, description=prop.get("description")))

    if not fields:
        fields["input"] = (str, Field(..., description="Free-form input"))
    return create_model(name, **fields)  # type: ignore


def _mcp_response_to_text(resp: Any) -> str:
    try:
        for c in getattr(resp, "content", []) or []:
            if getattr(c, "type", None) == "text" and getattr(c, "text", None):
                return c.text
    except Exception:
        pass
    return str(resp)


def _structured_call_factory(session, tool_name: str):
    async def _call(**kwargs):
        print(f"🔧 Tool call: {tool_name}, params: {kwargs}")
        response = await session.call_tool(tool_name, kwargs)

        query_id = current_query_id.get()

        return _mcp_response_to_text(response)


    return _call


async def create_tools_from_mcp(session) -> List[StructuredTool]:
    out: List[StructuredTool] = []
    result = await session.list_tools()

    for t in result.tools:
        tool_name = t.name  
        input_schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
        ArgsSchema = jsonschema_to_args_schema(f"{tool_name}Args", input_schema)

        call = _structured_call_factory(session, tool_name)  
        out.append(
            StructuredTool(
                name=tool_name,
                description=t.description or "MCP tool",
                args_schema=ArgsSchema,
                coroutine=call,  
            )
        )
        print(f"✅ Зарегистрирован StructuredTool: {tool_name}")
    return out

def group_tools_by_domain(tools: List[Tool]) -> Dict[AgentDomain, List[Tool]]:
    """Группировка инструментов по доменам"""
    tools_by_domain = {domain: [] for domain in AgentDomain}
    
    for tool in tools:
        domain = TOOL_DOMAINS.get(tool.name)
        if domain:
            tools_by_domain[domain].append(tool)
    
    return tools_by_domain


async def run_test_queries(orchestrator: OrchestratorAgent, queries: List[str]) -> None:
    """Запуск тестовых запросов"""
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"📝 Запрос {i}: {query}")
        print("="*70)
        
        try:
            result = await orchestrator.process_request(query)
            print(f"\n💬 Ответ: {result}")
        except Exception as e:
            print(f"\n❌ Ошибка при обработке запроса: {e}")
        
        print("-"*70)
        await asyncio.sleep(1)


async def run_interactive_mode(orchestrator: OrchestratorAgent) -> None:
    """Интерактивный режим общения"""
    print("\n" + "="*70)
    print("🎮 Интерактивный режим (введите 'exit' для выхода)")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
            if user_input.lower() in {'exit', 'quit', 'выход'} or '/Users/vanmac/finam-trader/.venv/bin/python /Users/vanmac/finam-trader/trader_mcp/main.py' in user_input:
                print("👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            result = await orchestrator.process_request(user_input)
            print(f"\n🤖 Ассистент: {result}")
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")



SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "mcp" / "server.py"
PYTHON_EXEC = sys.executable or "python"


def build_llm() -> ChatOpenAI:
    """Создает и возвращает настроенную модель LLM"""
    return ChatOpenAI(
        model=MODEL_ID,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
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
