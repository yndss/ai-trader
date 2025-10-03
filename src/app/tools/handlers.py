import datetime as dt
from typing import Any, Dict, List

from ..adapters import FinamClient
from ..viz.charts import candles_to_base64_png
from .schemas import (Account, CancelOrderInput, Candle, FindInstrumentInput,
                      GetCandlesInput, GetOrderbookInput, GetPortfolioInput,
                      PlaceOrderInput, Position)

finam = FinamClient()
PENDING_CONF: Dict[str, Dict[str, Any]] = {}

def get_accounts(_: dict) -> dict:
    return {"accounts":[a.model_dump() for a in finam.get_accounts()]}

def get_portfolio(args: dict) -> dict:
    data = GetPortfolioInput(**args)
    return {"positions":[p.model_dump() for p in finam.get_portfolio(data.account_id)]}

def find_instrument(args: dict) -> dict:
    data = FindInstrumentInput(**args)
    return {"instrument": finam.find_instrument(data.query)}

def get_candles(args: dict) -> dict:
    data = GetCandlesInput(**args)
    candles = finam.get_candles(data.figi, data.timeframe, data.start, data.end)
    img_b64 = candles_to_base64_png(candles)
    return {"candles":[c.model_dump() for c in candles], "plot_png_base64": img_b64}

def get_orderbook(args: dict) -> dict:
    data = GetOrderbookInput(**args)
    return finam.get_orderbook(data.figi, data.depth)

def place_order(args: dict) -> dict:
    data = PlaceOrderInput(**args)
    token = f"confirm_{int(dt.datetime.utcnow().timestamp()*1000)}"
    PENDING_CONF[token] = {"op":"place_order","args":data.model_dump()}
    return {"requires_confirmation": True, "confirm_token": token,
            "summary": f"{data.side.upper()} {data.qty} {data.figi} ({data.type})"}

def cancel_order(args: dict) -> dict:
    data = CancelOrderInput(**args)
    token = f"confirm_{int(dt.datetime.utcnow().timestamp()*1000)}"
    PENDING_CONF[token] = {"op":"cancel_order","args":data.model_dump()}
    return {"requires_confirmation": True, "confirm_token": token,
            "summary": f"Cancel order {data.order_id}"}

def confirm(token: str) -> dict:
    item = PENDING_CONF.pop(token, None)
    if not item:
        return {"status":"not_found"}
    if item["op"] == "place_order":
        a = item["args"]
        res = finam.place_order(a["account_id"], a["figi"], a["side"], a["qty"], a["type"], a.get("limit_price"))
    elif item["op"] == "cancel_order":
        a = item["args"]
        res = finam.cancel_order(a["account_id"], a["order_id"])
    else:
        res = {"error":"unknown operation"}
    return {"status":"executed","result":res}
        res = finam.place_order(a["account_id"], a["figi"], a["side"], a["qty"], a["type"], a.get("limit_price"))
    elif item["op"] == "cancel_order":
        a = item["args"]
        res = finam.cancel_order(a["account_id"], a["order_id"])
    else:
        res = {"error":"unknown operation"}
    return {"status":"executed","result":res}
