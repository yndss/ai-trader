from __future__ import annotations

"""Finam TradeAPI MCP server (REST)."""

import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TypeVar, Union

import httpx
from pydantic import BaseModel, Field, field_validator

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:
    raise RuntimeError("The 'mcp' package is required. Install it via 'pip install mcp'") from exc

try:
    from .models import Leg, Order, OrderType, Side, StopCondition, TimeInForce, ValidBefore
except ImportError:  # pragma: no cover - support running as a script
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.mcp.models import Leg, Order, OrderType, Side, StopCondition, TimeInForce, ValidBefore

FINAM_BASE_URL = os.getenv("FINAM_API_BASE_URL", "https://api.finam.ru")
DEFAULT_TIMEOUT = float(os.getenv("FINAM_HTTP_TIMEOUT", "20"))
DEFAULT_DEPTH = 10
ENV_SECRET = os.getenv("FINAM_ACCESS_TOKEN")


def _norm_symbol(symbol: str) -> str:
    normalized = (symbol or "").strip().upper()
    if "@" not in normalized:
        raise ValueError("symbol must be provided in format TICKER@MIC, e.g. SBER@MISX")
    return normalized


def _as_decimal(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    raise ValueError("numeric values must be int|float|str")


EnumT = TypeVar("EnumT", Side, OrderType, TimeInForce, StopCondition, ValidBefore)


def _parse_enum(enum_cls: type[EnumT], value: Union[str, EnumT, None]) -> Optional[EnumT]:
    if value is None or isinstance(value, enum_cls):
        return value

    raw = str(value).strip()
    normalised = raw.replace("-", "_").replace(" ", "_").upper()

    try:
        return enum_cls(normalised)
    except ValueError:
        parts = normalised.split("_")
        for idx in range(len(parts)):
            candidate = "_".join(parts[idx:])
            try:
                return enum_cls(candidate)
            except ValueError:
                continue
        raise ValueError(f"Unknown value '{value}' for enum {enum_cls.__name__}")


class GetBarsParams(BaseModel):
    symbol: str
    timeframe: str
    start_time: Optional[int] = Field(None, description="Unix seconds")
    end_time: Optional[int] = Field(None, description="Unix seconds")
    limit: Optional[int] = None

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return _norm_symbol(value)

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, value: str) -> str:
        return value.upper()


class AsyncFinamClient:
    def __init__(self, base_url: str = FINAM_BASE_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base = base_url.rstrip("/")
        self._jwt: Optional[str] = None
        self._secret: Optional[str] = ENV_SECRET
        self._client = httpx.AsyncClient(
            base_url=self._base,
            http2=True,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def set_jwt(self, jwt: Optional[str]) -> None:
        self._jwt = jwt

    def set_secret(self, secret: Optional[str]) -> None:
        self._secret = secret

    async def _auto_login(self, force: bool = False) -> bool:
        if not force and self._jwt:
            return True
        if not self._secret:
            return False

        try:
            result = await self.exchange_secret_for_jwt(self._secret)
        except Exception:
            return False

        token = result.get("jwt") or result.get("token")
        return bool(token)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        attach_auth: bool = True,
    ) -> Dict[str, Any]:
        if attach_auth and not self._jwt:
            await self._auto_login()

        response = None
        for attempt in range(2):
            headers: Dict[str, str] = {}
            if attach_auth and self._jwt:
                headers["Authorization"] = self._jwt

            response = await self._client.request(method, path, params=params, json=json, headers=headers)

            if response.status_code == 401 and attach_auth:
                refreshed = await self._auto_login(force=True)
                if refreshed:
                    continue
            break

        assert response is not None
        content_type = response.headers.get("content-type", "")
        ok = 200 <= response.status_code < 300

        if "application/json" in content_type:
            try:
                payload = response.json()
            except Exception:
                payload = {"raw": response.text}
        else:
            payload = {"raw": response.text}

        if not ok:
            if isinstance(payload, dict):
                payload.setdefault("ok", False)
                payload.setdefault("status", response.status_code)
            else:
                payload = {"ok": False, "status": response.status_code, "error": payload}
            return payload

        return payload if isinstance(payload, dict) else {"data": payload}

    async def exchange_secret_for_jwt(self, secret: Optional[str] = None) -> Dict[str, Any]:
        secret_to_use = secret or self._secret
        if not secret_to_use:
            raise RuntimeError("FINAM_ACCESS_TOKEN environment variable is not set and no secret provided")

        self.set_secret(secret_to_use)

        result = await self._request(
            "POST",
            "/v1/sessions",
            json={"secret": secret_to_use},
            attach_auth=False,
        )
        token = result.get("jwt") or result.get("token")
        if token:
            self.set_jwt(token)
        return result

    async def token_details(self, token: Optional[str] = None) -> Dict[str, Any]:
        payload = {"token": token or self._jwt}
        return await self._request("POST", "/v1/sessions/details", json=payload, attach_auth=False)

    async def get_account(self, account_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/v1/accounts/{account_id}")

    async def get_account_trades(
        self,
        account_id: str,
        *,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if start_time is not None:
            params["interval.start_time"] = start_time
        if end_time is not None:
            params["interval.end_time"] = end_time
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", f"/v1/accounts/{account_id}/trades", params=params or None)

    async def get_account_transactions(
        self,
        account_id: str,
        *,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if start_time is not None:
            params["interval.start_time"] = start_time
        if end_time is not None:
            params["interval.end_time"] = end_time
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", f"/v1/accounts/{account_id}/transactions", params=params or None)

    async def get_assets(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/assets")

    async def get_asset(self, symbol: str, account_id: Optional[str] = None) -> Dict[str, Any]:
        params = {"account_id": account_id} if account_id else None
        return await self._request("GET", f"/v1/assets/{_norm_symbol(symbol)}", params=params)

    async def get_asset_params(self, symbol: str, account_id: str) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/v1/assets/{_norm_symbol(symbol)}/params",
            params={"account_id": account_id},
        )

    async def get_options_chain(self, underlying_symbol: str) -> Dict[str, Any]:
        return await self._request("GET", f"/v1/assets/{_norm_symbol(underlying_symbol)}/options")

    async def get_asset_schedule(self, symbol: str) -> Dict[str, Any]:
        return await self._request("GET", f"/v1/assets/{_norm_symbol(symbol)}/schedule")

    async def get_clock(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/assets/clock")

    async def get_exchanges(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/exchanges")

    async def get_orders(self, account_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/v1/accounts/{account_id}/orders")

    async def get_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/v1/accounts/{account_id}/orders/{order_id}")

    async def place_order(self, order: Order) -> Dict[str, Any]:
        payload = order.to_request_payload()
        return await self._request("POST", f"/v1/accounts/{order.account_id}/orders", json=payload)

    async def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/v1/accounts/{account_id}/orders/{order_id}")

    async def last_quote(self, symbol: str) -> Dict[str, Any]:
        symbol = _norm_symbol(symbol)
        result = await self._request("GET", "/v1/marketdata/quotes/latest", params={"symbol": symbol})
        if result.get("ok") is False and result.get("status") == 404:
            result = await self._request("GET", f"/v1/instruments/{symbol}/quotes/latest")
        return result

    async def orderbook(self, symbol: str, depth: int = DEFAULT_DEPTH) -> Dict[str, Any]:
        symbol = _norm_symbol(symbol)
        params = {"symbol": symbol, "depth": depth}
        result = await self._request("GET", "/v1/marketdata/orderbook", params=params)
        if result.get("ok") is False and result.get("status") == 404:
            result = await self._request("GET", f"/v1/instruments/{symbol}/orderbook", params={"depth": depth})
        return result

    async def latest_trades(self, symbol: str) -> Dict[str, Any]:
        symbol = _norm_symbol(symbol)
        result = await self._request("GET", "/v1/marketdata/trades/latest", params={"symbol": symbol})
        if result.get("ok") is False and result.get("status") == 404:
            result = await self._request("GET", f"/v1/instruments/{symbol}/trades/latest")
        return result

    async def bars(self, params: GetBarsParams) -> Dict[str, Any]:
        query: Dict[str, Any] = {"symbol": params.symbol, "timeframe": params.timeframe}
        if params.start_time is not None:
            query["interval.start_time"] = params.start_time
        if params.end_time is not None:
            query["interval.end_time"] = params.end_time
        if params.limit is not None:
            query["limit"] = params.limit

        result = await self._request("GET", "/v1/marketdata/bars", params=query)
        if result.get("ok") is False and result.get("status") == 404:
            result = await self._request("GET", f"/v1/instruments/{params.symbol}/bars", params=query)
        return result


app = FastMCP("FinamTrader")
_client = AsyncFinamClient()


@app.tool()
async def get_jwt_token(secret: Optional[str] = None) -> Dict[str, Any]:
    return await _client.exchange_secret_for_jwt(secret)


@app.tool()
async def get_token_details(token: Optional[str] = None) -> Dict[str, Any]:
    return await _client.token_details(token)


@app.tool()
async def get_account(account_id: str) -> Dict[str, Any]:
    return await _client.get_account(account_id)


@app.tool()
async def get_account_trades(
    account_id: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    return await _client.get_account_trades(account_id, start_time=start_time, end_time=end_time, limit=limit)


@app.tool()
async def get_account_transactions(
    account_id: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    return await _client.get_account_transactions(account_id, start_time=start_time, end_time=end_time, limit=limit)


@app.tool()
async def list_accounts_via_token_details() -> Dict[str, Any]:
    details = await _client.token_details()
    if details.get("ok") is False:
        return details

    raw_ids: Iterable[Any] = details.get("account_ids") or []
    if not raw_ids and isinstance(details.get("accounts"), list):
        raw_ids = [
            acc.get("account_id") or acc.get("id")
            for acc in details["accounts"]
            if isinstance(acc, dict)
        ]

    account_ids = [account_id for account_id in raw_ids if account_id is not None]

    accounts: list[Dict[str, Any]] = []
    for account_id in account_ids:
        if account_id is None:
            continue
        account = await _client.get_account(str(account_id))
        accounts.append(account)

    return {"account_ids": account_ids, "accounts": accounts, "token_details": details}


@app.tool()
async def get_assets() -> Dict[str, Any]:
    return await _client.get_assets()


@app.tool()
async def get_asset(symbol: str, account_id: Optional[str] = None) -> Dict[str, Any]:
    return await _client.get_asset(symbol, account_id=account_id)


@app.tool()
async def get_asset_params(symbol: str, account_id: str) -> Dict[str, Any]:
    return await _client.get_asset_params(symbol, account_id)


@app.tool()
async def get_options_chain(underlying_symbol: str) -> Dict[str, Any]:
    return await _client.get_options_chain(underlying_symbol)


@app.tool()
async def get_asset_schedule(symbol: str) -> Dict[str, Any]:
    return await _client.get_asset_schedule(symbol)


@app.tool()
async def get_clock() -> Dict[str, Any]:
    return await _client.get_clock()


@app.tool()
async def get_exchanges() -> Dict[str, Any]:
    return await _client.get_exchanges()


@app.tool()
async def get_account_orders(account_id: str) -> Dict[str, Any]:
    return await _client.get_orders(account_id)


@app.tool()
async def get_order(account_id: str, order_id: str) -> Dict[str, Any]:
    return await _client.get_order(account_id, order_id)


@app.tool()
async def place_order(
    account_id: str,
    symbol: str,
    quantity: Union[str, int, float],
    side: Union[str, Side],
    order_type: Union[str, OrderType],
    time_in_force: Optional[Union[str, TimeInForce]] = None,
    limit_price: Optional[Union[str, int, float]] = None,
    stop_price: Optional[Union[str, int, float]] = None,
    stop_condition: Optional[Union[str, StopCondition]] = None,
    valid_before: Optional[Union[str, ValidBefore]] = None,
    client_order_id: Optional[str] = None,
    comment: Optional[str] = None,
    legs: Optional[Iterable[Union[Leg, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    legs_payload: Optional[list[Leg]] = None
    if legs is not None:
        legs_payload = []
        for leg in legs:
            if isinstance(leg, Leg):
                legs_payload.append(leg)
            else:
                legs_payload.append(Leg(**leg))

    order = Order(
        account_id=account_id,
        symbol=_norm_symbol(symbol),
        quantity=quantity,
        side=_parse_enum(Side, side),
        type=_parse_enum(OrderType, order_type),
        time_in_force=_parse_enum(TimeInForce, time_in_force),
        limit_price=_as_decimal(limit_price) if limit_price is not None else None,
        stop_price=_as_decimal(stop_price) if stop_price is not None else None,
        stop_condition=_parse_enum(StopCondition, stop_condition),
        valid_before=_parse_enum(ValidBefore, valid_before),
        client_order_id=client_order_id,
        comment=comment,
        legs=legs_payload,
    )
    return await _client.place_order(order)


@app.tool()
async def cancel_order(account_id: str, order_id: str) -> Dict[str, Any]:
    return await _client.cancel_order(account_id, order_id)


@app.tool()
async def get_last_quote(symbol: str) -> Dict[str, Any]:
    return await _client.last_quote(symbol)


@app.tool()
async def get_orderbook(symbol: str, depth: int = DEFAULT_DEPTH) -> Dict[str, Any]:
    return await _client.orderbook(symbol, depth=depth)


@app.tool()
async def get_latest_trades(symbol: str) -> Dict[str, Any]:
    return await _client.latest_trades(symbol)


@app.tool()
async def get_bars(
    symbol: str,
    timeframe: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    params = GetBarsParams(
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    return await _client.bars(params)

if __name__ == "__main__":
    try:
        app.run()
    finally:
        import asyncio

        asyncio.run(_client.aclose())
