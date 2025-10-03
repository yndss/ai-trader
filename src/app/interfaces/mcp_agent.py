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
from typing import Any, ClassVar, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools.base import BaseTool
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent
from langchain.schema import OutputParserException
from pydantic import PrivateAttr

load_dotenv()

try:
    from .call_logger import call_logger
except ImportError:  # pragma: no cover - fallback for standalone execution
    from call_logger import call_logger  # type: ignore


OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL_ID = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "mcp" / "server.py"
PYTHON_EXECUTABLE = sys.executable or "python"


class AgentDomain(Enum):
    """Supported functional domains for specialised agents."""

    AUTH = "auth"
    ACCOUNTS = "accounts"
    INSTRUMENTS = "instruments"
    ORDERS = "orders"
    MARKET_DATA = "market_data"


TOOL_DOMAINS: Dict[str, AgentDomain] = {
    "get_jwt_token": AgentDomain.AUTH,
    "get_token_details": AgentDomain.AUTH,
    "list_accounts_via_token_details": AgentDomain.ACCOUNTS,
    "get_account": AgentDomain.ACCOUNTS,
    "get_account_trades": AgentDomain.ACCOUNTS,
    "get_account_transactions": AgentDomain.ACCOUNTS,
    "get_assets": AgentDomain.INSTRUMENTS,
    "get_asset": AgentDomain.INSTRUMENTS,
    "get_asset_params": AgentDomain.INSTRUMENTS,
    "get_options_chain": AgentDomain.INSTRUMENTS,
    "get_asset_schedule": AgentDomain.INSTRUMENTS,
    "get_clock": AgentDomain.INSTRUMENTS,
    "get_exchanges": AgentDomain.INSTRUMENTS,
    "place_order": AgentDomain.ORDERS,
    "cancel_order": AgentDomain.ORDERS,
    "get_account_orders": AgentDomain.ORDERS,
    "get_order": AgentDomain.ORDERS,
    "get_last_quote": AgentDomain.MARKET_DATA,
    "get_orderbook": AgentDomain.MARKET_DATA,
    "get_latest_trades": AgentDomain.MARKET_DATA,
    "get_bars": AgentDomain.MARKET_DATA,
}


DOMAIN_DESCRIPTIONS: Dict[AgentDomain, str] = {
    AgentDomain.AUTH: "аутентификации и управления токенами",
    AgentDomain.ACCOUNTS: "работы со счетами, портфелями и балансами",
    AgentDomain.INSTRUMENTS: "поиска и анализа торговых инструментов",
    AgentDomain.ORDERS: "управления заявками (создание, отмена, мониторинг)",
    AgentDomain.MARKET_DATA: "получения рыночных данных (котировки, свечи, стаканы)",
}


class SpecializedAgent:
    """Wraps a LangChain agent configured for a specific functional domain."""

    def __init__(self, domain: AgentDomain, tools: List[BaseTool], llm: ChatOpenAI):
        self.domain = domain
        self.tools = tools
        self.llm = llm
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=3,
        )
        self.agent = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        """Create an agent with a domain-specific system prompt."""

        tool_descriptions = "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools)
        tool_names = ", ".join(tool.name for tool in self.tools)
        system_prompt = self._build_domain_prompt(tool_descriptions, tool_names)

        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=5,
            agent_kwargs={
                "system_prompt": system_prompt,
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
            },
        )

        # Ensure chat_history placeholder always resolves, even when empty.
        if hasattr(agent, "memory") and agent.memory:
            agent.memory.chat_memory.messages = agent.memory.chat_memory.messages or []
        if hasattr(agent, "agent") and hasattr(agent.agent, "llm_chain"):
            prompt_obj = getattr(agent.agent.llm_chain, "prompt", None)
            if prompt_obj is not None and hasattr(prompt_obj, "partial"):
                agent.agent.llm_chain.prompt = prompt_obj.partial(chat_history="")

        # Some LangChain versions ignore system_prompt; patch template directly if present.
        try:
            prompt = agent.agent.llm_chain.prompt  # type: ignore[attr-defined]
            if hasattr(prompt, "messages") and prompt.messages:
                first_message = prompt.messages[0]
                if hasattr(first_message, "prompt") and hasattr(first_message.prompt, "template"):
                    first_message.prompt.template = system_prompt
                elif hasattr(first_message, "content"):
                    first_message.content = system_prompt
        except Exception:  # pragma: no cover - defensive fallback
            pass

        parser = getattr(agent.agent, "output_parser", None)
        if parser is not None and not isinstance(parser, MCPOutputParser):
            agent.agent.output_parser = MCPOutputParser(parser)

        return agent

    def _build_domain_prompt(self, tools_desc: str, tool_names: str) -> str:
        """Построение промпта для специализированного агента"""
        return f"""Ты специализированный агент для {DOMAIN_DESCRIPTIONS[self.domain]}.

Доступные инструменты:
{tools_desc}

Используй JSON для вызова инструментов:
```json
{{{{
"action": $TOOL_NAME,
"action_input": $JSON_BLOB ("parametr_name": "value")
}}}}
```

Если инструмент не требует параметров, используй пустую строку в action_input.

Valid "action" values: "Final Answer" or one of [{tool_names}]

Формат работы:

Question: входной вопрос
Thought: анализ ситуации
Action:
$JSON_BLOB

Observation: результат действия

Action:
```json
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
- Отвечай на русском языке
- Форматируй ответы понятно и структурированно
- Если данных недостаточно, уточни у пользователя
- В случае получения любой ошибки выдавай Final Answer и сообщай пользователю об ошибке. ни за что не пробуй снова.

Thought:
"""

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
    """Routes user requests to the most relevant specialised agent."""

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

        lines: List[str] = []
        for message in history[-max_messages:]:
            role = "Пользователь" if getattr(message, "type", "human") == "human" else "Ассистент"
            content = (message.content or "")[:max_length]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def route_request(self, user_input: str) -> AgentDomain:
        history_snapshot = self._get_history()
        routing_prompt = (
            "Ты агент-маршрутизатор для системы Finam.\n\n"
            "Доступные агенты: AUTH, ACCOUNTS, INSTRUMENTS, ORDERS, MARKET_DATA.\n"
            f"История диалога:\n{history_snapshot}\n\n"
            f"Запрос пользователя: {user_input}\n\n"
            "Ответь ОДНИМ словом из списка выше."
        )

        response = await self.llm.ainvoke(routing_prompt)
        content = getattr(response, "content", "")
        domain_key = str(content).strip().upper()
        domain = self.DOMAIN_MAP.get(domain_key, AgentDomain.ACCOUNTS)
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


class MCPAsyncTool(BaseTool):
    """LangChain tool wrapper around an MCP server method."""

    args_schema: ClassVar[Optional[Any]] = None
    _session: ClientSession = PrivateAttr()

    def __init__(self, session: ClientSession, tool_name: str, description: Optional[str] = None):
        super().__init__(name=tool_name, description=description or "MCP tool")
        self._session = session

    def _run(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - sync mode unused
        raise NotImplementedError("Используйте асинхронный режим для MCP инструментов")

    @staticmethod
    def _normalize_params(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        for arg in args:
            if isinstance(arg, dict):
                params.update(arg)
            elif arg is not None:
                params.setdefault("input", arg)

        params.update(kwargs)

        for key in ("tool_input", "input", "args"):
            if key not in params:
                continue
            value = params[key]
            if isinstance(value, dict):
                params.pop(key)
                params.update(value)
            elif isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    params.pop(key)
                    params.update(parsed)

        return params

    @staticmethod
    def _format_tool_result(result: CallToolResult) -> str:
        chunks: List[str] = []
        if result.structuredContent:
            chunks.append(json.dumps(result.structuredContent, ensure_ascii=False))

        for item in result.content:
            if isinstance(item, TextContent):
                chunks.append(item.text)
            else:
                chunks.append(item.model_dump_json())

        return "\n".join(chunk for chunk in chunks if chunk).strip()

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        kwargs.pop("config", None)
        kwargs.pop("run_manager", None)

        params = self._normalize_params(args, kwargs)
        print(f"🔧 Tool call: {self.name}, params: {params}")

        try:
            call_logger.log_tool_call(self.name, params)
        except Exception as log_error:  # pragma: no cover - logging is best-effort
            print(f"⚠️  Не удалось записать вызов инструмента {self.name}: {log_error}")

        response = await self._session.call_tool(self.name, params)

        if response.isError:
            details = self._format_tool_result(response)
            return f"Ошибка при вызове {self.name}: {details}"

        return self._format_tool_result(response)


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
                # Preserve already quoted values
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


async def create_tools_from_mcp(session: ClientSession) -> List[BaseTool]:
    tools: List[BaseTool] = []
    cursor: Optional[str] = None

    try:
        while True:
            listing = await session.list_tools(cursor=cursor)
            for tool in listing.tools:
                description = tool.description or tool.title or "MCP tool"
                tools.append(MCPAsyncTool(session, tool.name, description))

            cursor = listing.nextCursor
            if not cursor:
                break
    except Exception as exc:
        print(f"❌ Ошибка при получении инструментов MCP: {exc}")
        return []

    return tools


def group_tools_by_domain(tools: Iterable[BaseTool]) -> Dict[AgentDomain, List[BaseTool]]:
    grouped: Dict[AgentDomain, List[BaseTool]] = {domain: [] for domain in AgentDomain}
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
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY не установлен в окружении")

    return ChatOpenAI(
        model=OPENROUTER_MODEL_ID,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )


async def main(interactive: bool | None = None, test_queries: Optional[List[str]] = None) -> None:
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

            if interactive is None and test_queries is None:
                answer = input("\n🎮 Запустить интерактивный режим? (y/n): ").strip().lower()
                interactive = answer == "y"

            if test_queries:
                await run_test_queries(orchestrator, test_queries)

            if interactive or (interactive is None and test_queries is None):
                await run_interactive_mode(orchestrator)


def main_cli() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До свидания!")


if __name__ == "__main__":
    main_cli()
