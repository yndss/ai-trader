from __future__ import annotations

"""LangChain-powered multi-agent orchestrator over the Finam MCP server."""

import asyncio
import ast
import json
import os
import sys
import traceback
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Type

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, create_model

try:
    from .call_logger import call_logger
except ImportError:  # pragma: no cover - fallback for standalone execution
    from call_logger import call_logger  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


def _env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


DEFAULT_ACCOUNT_ID = os.getenv("DEFAULT_ACCOUNT_ID", "TRQD05:409933")

OPENROUTER_API_KEY = _env_value("OPENROUTER_API_KEY", "COMET_API_KEY", "LLM_API_KEY")
OPENROUTER_BASE_URL = _env_value("OPENROUTER_BASE", "COMET_BASE_URL", "LLM_BASE_URL")
OPENROUTER_MODEL_ID = _env_value(
    "OPENROUTER_MODEL",
    "COMET_MODEL_ID",
    "LLM_MODEL_ID",
    "LLM_MODEL",
)

SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "mcp" / "server.py"
PYTHON_EXECUTABLE = sys.executable or "python"


class AgentDomain(Enum):
    """Домены специализированных агентов."""

    ACCOUNTS = "accounts"
    INSTRUMENTS = "instruments"
    ORDERS = "orders"
    MARKET_DATA = "market_data"
    AUTH = "auth"


TOOL_DOMAINS: Dict[str, AgentDomain] = {
    # Auth domain
    "Auth": AgentDomain.AUTH,
    "TokenDetails": AgentDomain.AUTH,

    # Accounts domain
    "GetAccount": AgentDomain.ACCOUNTS,
    "Trades": AgentDomain.ACCOUNTS,
    "Transactions": AgentDomain.ACCOUNTS,

    # Instruments domain
    "GetAssets": AgentDomain.INSTRUMENTS,
    "GetAsset": AgentDomain.INSTRUMENTS,
    "GetAssetParams": AgentDomain.INSTRUMENTS,
    "OptionsChain": AgentDomain.INSTRUMENTS,
    "Schedule": AgentDomain.INSTRUMENTS,
    "Clock": AgentDomain.INSTRUMENTS,
    "Exchanges": AgentDomain.INSTRUMENTS,

    # Orders domain
    "PlaceOrder": AgentDomain.ORDERS,
    "GetOrders": AgentDomain.ORDERS,
    "GetOrder": AgentDomain.ORDERS,
    "CancelOrder": AgentDomain.ORDERS,

    # Market Data domain
    "Bars": AgentDomain.MARKET_DATA,
    "LastQuote": AgentDomain.MARKET_DATA,
    "LatestTrades": AgentDomain.MARKET_DATA,
    "OrderBook": AgentDomain.MARKET_DATA,
}


DOMAIN_DESCRIPTIONS = {
    AgentDomain.AUTH: "аутентификации и управления токенами",
    AgentDomain.ACCOUNTS: "работы со счетами, портфелями и балансами",
    AgentDomain.INSTRUMENTS: "поиска и анализа торговых инструментов",
    AgentDomain.ORDERS: "управления заявками (создание, отмена, мониторинг)",
    AgentDomain.MARKET_DATA: "получения рыночных данных (котировки, свечи, стаканы)",
}


class SpecializedAgent:
    """Специализированный агент для конкретного домена."""

    def __init__(self, domain: AgentDomain, tools: List[StructuredTool], llm: ChatOpenAI):
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

    def _create_agent(self) -> AgentExecutor:
        tool_names = ", ".join(tool.name for tool in self.tools)
        tools_desc = "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools)
        system_prompt = self._build_domain_prompt(tools_desc, tool_names)

        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
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

        # Подменяем системный промпт на доменно-специализированный.
        prompt = getattr(agent.agent.llm_chain, "prompt", None)
        if prompt is not None and getattr(prompt, "messages", None):
            first_message = prompt.messages[0]
            if hasattr(first_message, "prompt") and hasattr(first_message.prompt, "template"):
                first_message.prompt.template = system_prompt
            elif hasattr(first_message, "content"):
                first_message.content = system_prompt
            input_variables = getattr(prompt, "input_variables", None)
            if isinstance(input_variables, list) and "chat_history" not in input_variables:
                input_variables.append("chat_history")

        parser = getattr(agent.agent, "output_parser", None)
        if parser is not None and not isinstance(parser, MCPOutputParser):
            agent.agent.output_parser = MCPOutputParser(parser)

        return agent

    def _build_domain_prompt(self, tools_desc: str, tool_names: str) -> str:
        return dedent(
            f"""
            Ты специализированный агент для {DOMAIN_DESCRIPTIONS[self.domain]}.

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
            - Отвечай ТОЛЬКО на вопросы в твоей области ({DOMAIN_DESCRIPTIONS[self.domain]})
            - Всегда используй инструменты для получения актуальных данных
            - ВСЕГДА ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
            - Форматируй ответы понятно и структурированно
            - Если данных недостаточно, уточни у пользователя
            - В случае любой ошибки сразу выдай Final Answer и сообщи пользователю об ошибке. Ни за что не повторяй один и тот же запрос повторно.
            - Если не указан ID аккаунта, используй значение по умолчанию: {DEFAULT_ACCOUNT_ID}
            - ЕСЛИ ТЕБЕ НЕ ХВАТАЕТ ИНФОРМАЦИИ — УЗНАЙ ЕЁ С ПОМОЩЬЮ ИНСТРУМЕНТОВ. НИКОГДА НЕ ПРЕДПОЛАГАЙ.

            Thought:
            """
        ).strip()

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        task_input = task
        if context and context.get("global_history"):
            task_input = f"Контекст:\n{context['global_history']}\n\nЗапрос: {task}"

        call_logger.clear_question_history(task)
        token = call_logger.set_current_question(task)
        try:
            result = await self.agent.ainvoke({"input": task_input})
        except Exception as exc:  # pylint: disable=broad-except
            print("⚠️  SpecializedAgent: ошибка выполнения агента.")
            print("   ↳ домен:", self.domain.value)
            print("   ↳ входной запрос:\n", task_input)
            print("   ↳ тип исключения:", repr(exc))
            print("   ↳ traceback:\n", traceback.format_exc())
            history = call_logger.question_history(task)
            if history:
                print("   ↳ вызовы инструментов:")
                print(json.dumps(history, ensure_ascii=False, indent=2))
            else:
                print("   ↳ инструменты не вызывались")
            raise
        finally:
            call_logger.reset_current_question(token)

        return result.get("output", str(result))


class OrchestratorAgent:
    """Оркестратор для маршрутизации запросов между агентами."""

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
            k=10,
        )

    def add_agent(self, agent: SpecializedAgent) -> None:
        self.specialized_agents[agent.domain] = agent

    def _get_history(self, max_messages: int = 6, max_length: int = 200) -> str:
        memory_vars = self.global_memory.load_memory_variables({})
        history = memory_vars.get("chat_history") or []
        if not history:
            return "Нет предыдущих сообщений"

        result: List[str] = []
        for message in history[-max_messages:]:
            role = "Пользователь" if getattr(message, "type", "human") == "human" else "Ассистент"
            content = (message.content or "")[:max_length]
            result.append(f"{role}: {content}")
        return "\n".join(result)

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
            {self._get_history()}

            Запрос пользователя: {user_input}

            Ответь ТОЛЬКО одним словом из списка: AUTH, ACCOUNTS, INSTRUMENTS, ORDERS, MARKET_DATA.
            """
        ).strip()

        response = await self.llm.ainvoke(routing_prompt)
        content = getattr(response, "content", "")
        domain = self.DOMAIN_MAP.get(str(content).strip().upper(), AgentDomain.ACCOUNTS)
        print(f"\n🎯 Оркестратор направил запрос агенту: {domain.value}")
        return domain

    async def process_request(self, user_input: str) -> str:
        self.global_memory.chat_memory.add_user_message(user_input)
        try:
            domain = await self.route_request(user_input)
            agent = self.specialized_agents.get(domain)
            if agent is None:
                message = f"Агент для домена {domain.value} не найден"
                self.global_memory.chat_memory.add_ai_message(message)
                return message

            context = {"global_history": self._get_history()}
            result = await agent.execute(user_input, context)
            self.global_memory.chat_memory.add_ai_message(result)
            return result
        except Exception as exc:
            error_msg = f"Произошла ошибка при обработке запроса: {exc}"
            print(f"❌ Ошибка: {error_msg}")
            self.global_memory.chat_memory.add_ai_message(error_msg)
            return error_msg


_JSON_TO_PY: Dict[str, Type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def jsonschema_to_args_schema(name: str, schema: Optional[Dict[str, Any]]) -> Type[BaseModel]:
    schema = schema or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: Dict[str, tuple[type[Any], Field[Any]]] = {}

    for key, prop in properties.items():
        json_type = prop.get("type", "string")
        py_type = _JSON_TO_PY.get(json_type, str)
        default = ... if key in required else None
        fields[key] = (py_type, Field(default, description=prop.get("description")))

    if not fields:
        fields["input"] = (str, Field(..., description="Free-form input"))

    return create_model(name, **fields)  # type: ignore[return-value]


def _mcp_response_to_text(response: Any) -> str:
    try:
        for content in getattr(response, "content", []) or []:
            if getattr(content, "type", None) == "text" and getattr(content, "text", None):
                return content.text
    except Exception:  # pragma: no cover - best effort fallback
        pass
    return str(response)


def _structured_call_factory(session: ClientSession, tool_name: str):
    async def _call(**kwargs: Any) -> str:
        params = dict(kwargs)
        params.setdefault("account_id", DEFAULT_ACCOUNT_ID)

        try:
            call_logger.log_tool_call(tool_name, params)
        except Exception as log_exc:  # pragma: no cover - logging best effort
            print(f"⚠️  Не удалось записать вызов инструмента {tool_name}: {log_exc}")

        response = await session.call_tool(tool_name, params)
        if getattr(response, "isError", False):
            details = _mcp_response_to_text(response)
            return f"Ошибка при вызове {tool_name}: {details}"
        return _mcp_response_to_text(response)

    return _call


async def create_tools_from_mcp(session: ClientSession) -> List[StructuredTool]:
    tools: List[StructuredTool] = []
    cursor: Optional[str] = None

    while True:
        listing = await session.list_tools(cursor=cursor)
        for tool in listing.tools:
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            args_schema = jsonschema_to_args_schema(f"{tool.name}Args", schema)
            coroutine = _structured_call_factory(session, tool.name)
            tools.append(
                StructuredTool(
                    name=tool.name,
                    description=tool.description or tool.title or "MCP tool",
                    args_schema=args_schema,
                    coroutine=coroutine,
                )
            )
            print(f"✅ Зарегистрирован StructuredTool: {tool.name}")

        cursor = getattr(listing, "nextCursor", None)
        if not cursor:
            break

    return tools


def group_tools_by_domain(tools: Iterable[StructuredTool]) -> Dict[AgentDomain, List[StructuredTool]]:
    grouped: Dict[AgentDomain, List[StructuredTool]] = {domain: [] for domain in AgentDomain}
    for tool in tools:
        domain = TOOL_DOMAINS.get(tool.name, AgentDomain.ACCOUNTS)
        grouped.setdefault(domain, []).append(tool)
    return grouped


async def run_test_queries(orchestrator: OrchestratorAgent, queries: Iterable[str]) -> None:
    for idx, query in enumerate(queries, start=1):
        print("\n" + "=" * 70)
        print(f"📝 Запрос {idx}: {query}")
        print("=" * 70)
        result = await orchestrator.process_request(query)
        print(f"\n💬 Ответ: {result}\n" + "-" * 70)
        await asyncio.sleep(0.5)


async def run_interactive_mode(orchestrator: OrchestratorAgent) -> None:
    print("\n" + "=" * 70)
    print("🎮 Интерактивный режим (введите 'exit' для выхода)")
    print("=" * 70)

    loop = asyncio.get_running_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(None, input, "\n👤 Вы: ")
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            return

        if user_input.strip().lower() in {"exit", "quit", "выход"}:
            print("👋 До свидания!")
            return

        if not user_input.strip():
            continue

        response = await orchestrator.process_request(user_input)
        print(f"\n🤖 Ассистент: {response}")


def build_llm() -> ChatOpenAI:
    missing: List[str] = []
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY/COMET_API_KEY")
    if not OPENROUTER_BASE_URL:
        missing.append("OPENROUTER_BASE/COMET_BASE_URL")
    if not OPENROUTER_MODEL_ID:
        missing.append("OPENROUTER_MODEL/COMET_MODEL_ID")

    if missing:
        raise RuntimeError(
            "Не установлены переменные окружения: " + ", ".join(missing)
        )

    return ChatOpenAI(
        model=OPENROUTER_MODEL_ID,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )


async def main(
    interactive: bool | None = None,
    test_queries: Optional[List[str]] = None,
) -> None:
    if not SERVER_SCRIPT.exists():
        raise FileNotFoundError(f"Не найден MCP сервер по пути {SERVER_SCRIPT}")

    llm = build_llm()

    server_params = StdioServerParameters(
        command=PYTHON_EXECUTABLE,
        args=[str(SERVER_SCRIPT)],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await create_tools_from_mcp(session)
            if not tools:
                print("❌ Не удалось загрузить инструменты MCP")
                return

            tools_by_domain = group_tools_by_domain(tools)
            orchestrator = OrchestratorAgent(llm)

            for domain, domain_tools in tools_by_domain.items():
                if not domain_tools:
                    continue
                agent = SpecializedAgent(domain, domain_tools, llm)
                orchestrator.add_agent(agent)
                print(f"✅ Создан агент '{domain.value}' с {len(domain_tools)} инструментами")

            if test_queries:
                await run_test_queries(orchestrator, test_queries)

            if interactive is None and test_queries is None:
                answer = input("\n🎮 Запустить интерактивный режим? (y/n): ").strip().lower()
                interactive = answer == "y"

            if interactive or (interactive is None and not test_queries):
                await run_interactive_mode(orchestrator)


def main_cli() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До свидания!")


class MCPOutputParser:
    """Wrap LangChain structured chat parser to auto-fix minor JSON formatting issues."""

    def __init__(self, inner_parser):
        self._inner = inner_parser

    def parse(self, text: str):  # type: ignore[override]
        try:
            return self._inner.parse(text)
        except Exception as exc:  # pylint: disable=broad-except
            print("⚠️  MCPOutputParser: не удалось распарсить ответ, пробуем восстановить.")
            print("   ↳ исходный ответ модели:\n", text)
            print("   ↳ тип исключения:", repr(exc))
            repaired = self._repair_action_block(text)
            if repaired != text:
                print("   ↳ после попытки восстановления:\n", repaired)
            if repaired == text:
                print("⚠️  MCPOutputParser: не удалось распознать Action блок:\n", text)
                raise
            try:
                return self._inner.parse(repaired)
            except Exception as second_exc:  # pylint: disable=broad-except
                print("⚠️  MCPOutputParser: исправление не помогло — повторный сбой.")
                print("   ↳ восстановленный текст:\n", repaired)
                print("   ↳ тип исключения:", repr(second_exc))
                raise

    async def aparse(self, text: str):  # type: ignore[override]
        async def _call_async(target, payload):
            if hasattr(target, "aparse"):
                return await target.aparse(payload)
            return target.parse(payload)

        try:
            return await _call_async(self._inner, text)
        except Exception as exc:  # pylint: disable=broad-except
            print("⚠️  MCPOutputParser: aparse не справился, пробуем синхронную починку.")
            print("   ↳ исходный ответ модели:\n", text)
            print("   ↳ тип исключения:", repr(exc))
            repaired = self._repair_action_block(text)
            if repaired != text:
                print("   ↳ после попытки восстановления:\n", repaired)
            if repaired == text:
                raise
            return await _call_async(self._inner, repaired)

    def get_format_instructions(self) -> str:
        return self._inner.get_format_instructions()

    @staticmethod
    def _repair_action_block(text: str) -> str:
        import re

        pattern = re.compile(r"Action:\s*(?P<body>\{?.*?)(\nObservation:|\Z)", re.DOTALL)

        def _strip_code_fence(value: str) -> str:
            cleaned = value.strip()
            if not cleaned.startswith("```"):
                return cleaned

            fence_free = cleaned[3:]
            fence_free = fence_free.lstrip().removeprefix("json").removeprefix("JSON")
            if "```" in fence_free:
                fence_free = fence_free.split("```", 1)[0]
            return fence_free.strip()

        def _quote_json_keys(payload: str) -> str:
            import re as _re

            return _re.sub(r"(?<![\"'])\b([A-Za-z_][\w]*)\b(?=\s*:)", r'"\1"', payload)

        def _quote_bare_values(payload: str) -> str:
            import re as _re

            def _replacer(match: _re.Match[str]) -> str:
                prefix = match.group(1)
                value = match.group(2)
                if value.startswith(('"', "'", "{", "[")):
                    return prefix + value
                return prefix + f'"{value.strip()}"'

            patterns = [
                r'("action"\s*:\s*)([^\s",}][^",}]*)',
                r'("tool"\s*:\s*)([^\s",}][^",}]*)',
                r'("name"\s*:\s*)([^\s",}][^",}]*)',
            ]

            updated = payload
            for pattern in patterns:
                updated = _re.sub(pattern, _replacer, updated)
            return updated

        def _candidate_payloads(body: str) -> List[str]:
            stripped = _strip_code_fence(body)
            if not stripped:
                return []

            raw = stripped.strip()
            candidates: List[str] = []

            variants = {raw}
            variants.add(raw.strip("{} \n"))
            variants.update({raw.replace("'", '"'), raw.strip("{} \n").replace("'", '"')})

            normalized: set[str] = set()
            for variant in variants:
                trimmed = variant.strip()
                if not trimmed:
                    continue
                normalized.add(trimmed)
                normalized.add(_quote_json_keys(trimmed))
                normalized.add(_quote_bare_values(_quote_json_keys(trimmed)))

            for variant in normalized:
                candidate = variant.strip()
                if not candidate:
                    continue
                if not candidate.startswith("{"):
                    if ":" not in candidate:
                        action_value = candidate.strip().strip('\"')
                        candidates.append(json.dumps({"action": action_value}))
                        candidates.append(json.dumps({"action": action_value, "action_input": {}}))
                        continue
                    candidate = "{" + candidate.strip("{} ") + "}"
                candidates.append(candidate)

            return candidates

        def _safe_load(payload: str) -> Optional[Dict[str, Any]]:
            try:
                loaded = json.loads(payload)
                if isinstance(loaded, dict):
                    return loaded
            except json.JSONDecodeError:
                pass

            try:
                loaded = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                return None

            return loaded if isinstance(loaded, dict) else None

        def _normalize_action_dict(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(data, dict):
                return None

            normalized: Dict[str, Any] = dict(data)

            aliases = {
                "tool": "action",
                "Tool": "action",
                "tool_name": "action",
                "ToolName": "action",
                "toolName": "action",
            }
            for old_key, new_key in aliases.items():
                if old_key in normalized and new_key not in normalized:
                    normalized[new_key] = normalized.pop(old_key)

            input_aliases = {
                "tool_input": "action_input",
                "Tool Input": "action_input",
                "toolInput": "action_input",
                "ToolInput": "action_input",
                "args": "action_input",
                "Arguments": "action_input",
                "arguments": "action_input",
                "params": "action_input",
                "parameters": "action_input",
                "input": "action_input",
                "Action Input": "action_input",
                "actionInput": "action_input",
            }
            for old_key, new_key in input_aliases.items():
                if old_key in normalized and new_key not in normalized:
                    normalized[new_key] = normalized.pop(old_key)

            action_value = normalized.get("action")
            if isinstance(action_value, dict):
                nested_name = (
                    action_value.get("action")
                    or action_value.get("tool")
                    or action_value.get("name")
                )
                if isinstance(nested_name, str):
                    normalized["action"] = nested_name

            if "action" not in normalized or not isinstance(normalized["action"], str):
                return None

            action_input = normalized.get("action_input")
            if isinstance(action_input, str) and normalized["action"] != "Final Answer":
                potential = _safe_load(action_input)
                if potential is not None:
                    normalized["action_input"] = potential
                else:
                    normalized["action_input"] = {"value": action_input}
            elif action_input is None and normalized["action"] != "Final Answer":
                normalized["action_input"] = {}

            return normalized

        def _repair_single_match(match: re.Match[str]) -> str:
            body = match.group("body")
            for payload in _candidate_payloads(body):
                parsed = _safe_load(payload)
                if parsed is None:
                    continue
                normalized = _normalize_action_dict(parsed)
                if normalized is None:
                    continue
                fixed_body = json.dumps(normalized, ensure_ascii=False)
                original = match.group(0)
                return original.replace(body, fixed_body, 1)
            return match.group(0)

        return pattern.sub(_repair_single_match, text)


if __name__ == "__main__":
    main_cli()
