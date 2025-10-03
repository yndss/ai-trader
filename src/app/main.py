from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .llm import call_llm_with_tools
from .tools import TOOL_IMPL, TOOLS_SPEC
from .tools.registry import confirm_operation

app = FastAPI(title="Trader AI Assistant (Finam)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

SYSTEM_PROMPT = (
    "You are a Russian-speaking trading assistant. "
    "You MUST use tools for any financial or portfolio data. "
    "For place/cancel order, ALWAYS ask for explicit confirmation (return requires_confirmation). "
    "Be concise and helpful."
)

class ChatInput(BaseModel):
    user: str
    account_id: str | None = None

class ConfirmInput(BaseModel):
    confirm_token: str

@app.post("/chat")
def chat(inp: ChatInput) -> Dict[str, Any]:
    msgs = [{"role":"system","content": SYSTEM_PROMPT}, {"role":"user","content": inp.user}]
    llm = call_llm_with_tools(msgs, TOOLS_SPEC)
    choice = llm["choices"][0]["message"]
    results = []
    tool_calls = choice.get("tool_calls") or []
    if tool_calls:
        import json
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"].get("arguments") or "{}")
            out = TOOL_IMPL[name](args)
            results.append({"tool_name": name, "result": out})
        return {"assistant_text": choice.get("content") or "", "tool_results": results}
    return {"assistant_text": choice.get("content") or "", "tool_results": []}

@app.post("/confirm")
def confirm(inp: ConfirmInput) -> Dict[str, Any]:
    return confirm_operation(inp.confirm_token)
def confirm(inp: ConfirmInput) -> Dict[str, Any]:
    return confirm_operation(inp.confirm_token)
