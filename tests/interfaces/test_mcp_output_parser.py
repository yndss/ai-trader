import json
import re

from src.app.interfaces.mcp_agent import MCPOutputParser


def _extract_action_payload(text: str) -> dict:
    match = re.search(r"Action:\s*(\{.*?\})(?:\nObservation:|\Z)", text, flags=re.DOTALL)
    assert match, f"Не удалось найти JSON блок в тексте:\n{text}"
    return json.loads(match.group(1))


def test_repair_action_block_converts_tool_aliases() -> None:
    raw = (
        "Thought: проверяем баланс\n"
        "Action:\n"
        '"tool": "get_account",\n'
        '"tool_input": {"account_id": "A1"}\n'
        "Observation: информация получена"
    )

    repaired = MCPOutputParser._repair_action_block(raw)
    payload = _extract_action_payload(repaired)

    assert payload == {"action": "get_account", "action_input": {"account_id": "A1"}}


def test_repair_handles_code_fence_and_single_quotes() -> None:
    raw = (
        "Action:\n"
        "```json\n"
        "{\n"
        "  'toolName': 'get_account',\n"
        "  \"args\": {'id': 'B2'}\n"
        "}\n"
        "```\n"
        "Observation: ок"
    )

    repaired = MCPOutputParser._repair_action_block(raw)
    payload = _extract_action_payload(repaired)

    assert payload == {"action": "get_account", "action_input": {"id": "B2"}}


def test_repair_preserves_final_answer_without_action_input() -> None:
    raw = "Action:\nFinal Answer\n"

    repaired = MCPOutputParser._repair_action_block(raw)
    payload = _extract_action_payload(repaired)

    assert payload == {"action": "Final Answer"}


def test_repair_parses_embedded_json_string_inputs() -> None:
    raw = (
        "Action:\n"
        "{\n"
        "  \"tool\": \"get_account\",\n"
        "  \"tool_input\": \"{\\\"account_id\\\": \\\"C3\\\"}\"\n"
        "}\n"
        "Observation: готово"
    )

    repaired = MCPOutputParser._repair_action_block(raw)
    payload = _extract_action_payload(repaired)

    assert payload == {"action": "get_account", "action_input": {"account_id": "C3"}}


def test_repair_quotes_tool_value_without_quotes() -> None:
    raw = (
        "Action:\n"
        "{\n"
        "  action: get_account,\n"
        "  action_input: {account_id: D4}\n"
        "}\n"
        "Observation: сделано"
    )

    repaired = MCPOutputParser._repair_action_block(raw)
    payload = _extract_action_payload(repaired)

    assert payload == {"action": "get_account", "action_input": {"account_id": "D4"}}
