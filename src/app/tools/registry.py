from typing import Dict, Any
from . import handlers as H

TOOLS_SPEC = [
    {"type":"function","function":{"name":"get_accounts","description":"List trading accounts.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"get_portfolio","description":"Portfolio positions for account.","parameters":{"type":"object","properties":{"account_id":{"type":"string"}},"required":["account_id"]}}},
    {"type":"function","function":{"name":"find_instrument","description":"Find instrument by ticker/name.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"get_candles","description":"Historical candles.","parameters":{"type":"object","properties":{"figi":{"type":"string"},"timeframe":{"type":"string","enum":["1m","5m","15m","1h","1d"]},"start":{"type":"string","format":"date-time"},"end":{"type":"string","format":"date-time"}},"required":["figi","start","end"]}}},
    {"type":"function","function":{"name":"get_orderbook","description":"Orderbook snapshot.","parameters":{"type":"object","properties":{"figi":{"type":"string"},"depth":{"type":"integer","minimum":1,"maximum":50}},"required":["figi"]}}},
    {"type":"function","function":{"name":"place_order","description":"Place order (requires confirmation).","parameters":{"type":"object","properties":{"account_id":{"type":"string"},"figi":{"type":"string"},"side":{"type":"string","enum":["buy","sell"]},"qty":{"type":"number"},"type":{"type":"string","enum":["market","limit"]},"limit_price":{"type":["number","null"]}},"required":["account_id","figi","side","qty"]}}}},
    {"type":"function","function":{"name":"cancel_order","description":"Cancel order (requires confirmation).","parameters":{"type":"object","properties":{"account_id":{"type":"string"},"order_id":{"type":"string"}},"required":["account_id","order_id"]}}}},
]

TOOL_IMPL: Dict[str, Any] = {
    "get_accounts": H.get_accounts,
    "get_portfolio": H.get_portfolio,
    "find_instrument": H.find_instrument,
    "get_candles": H.get_candles,
    "get_orderbook": H.get_orderbook,
    "place_order": H.place_order,
    "cancel_order": H.cancel_order,
}

confirm_operation = H.confirm
