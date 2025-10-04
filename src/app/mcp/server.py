from __future__ import annotations

"""FastMCP server exposing Finam TradeAPI endpoints as tools."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

try:
    from finam_client import FinamAPIClient
except ImportError:  # pragma: no cover - fallback if namespace import fails
    from finam_client.client import FinamAPIClient  # type: ignore

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

if load_dotenv is not None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


mcp = FastMCP("FinamTrader")
api_client = FinamAPIClient()

_DEFAULT_SECRET = os.getenv("FINAM_AUTH_SECRET") or os.getenv("FINAM_ACCESS_TOKEN") or ""
_CURRENT_TOKEN: Optional[str] = None


def _set_authorization(token: Optional[str]) -> None:
    global _CURRENT_TOKEN
    if token:
        formatted = token.strip()
        api_client.session.headers["Authorization"] = formatted
        _CURRENT_TOKEN = formatted
    else:
        api_client.session.headers.pop("Authorization", None)
        _CURRENT_TOKEN = None


_initial_auth = api_client.session.headers.get("Authorization")
if _initial_auth:
    _set_authorization(_initial_auth)


def _exchange_secret_for_token(secret: str) -> Dict[str, Any]:
    global _DEFAULT_SECRET
    if secret:
        _DEFAULT_SECRET = secret
    response = api_client.execute_request(
        "POST",
        "/v1/sessions",
        json={"secret": secret},
    )

    token = response.get("token") or response.get("jwt")
    if token:
        _set_authorization(token)
    return response


async def _ensure_authorized() -> None:
    current_header = api_client.session.headers.get("Authorization")
    if current_header:
        if _DEFAULT_SECRET and current_header.strip() == _DEFAULT_SECRET.strip():
            pass
        elif _CURRENT_TOKEN and (_DEFAULT_SECRET == "" or _CURRENT_TOKEN.strip() != _DEFAULT_SECRET.strip()):
            return
    if _DEFAULT_SECRET:
        _exchange_secret_for_token(_DEFAULT_SECRET)


@mcp.tool()
async def Auth(secret: str) -> Dict[str, Any]:
    """
    Get JWT token from API token

    Args:
        secret: API token (secret key)

    Returns:
        dict: JWT token information with the following structure:
            - token (str): Received JWT token
    """
    response = _exchange_secret_for_token(secret)
    return response


@mcp.tool()
async def TokenDetails(token: str) -> Dict[str, Any]:
    """
    Get information about session token

    Args:
        token: JWT token

    Returns:
        dict: Token information with the following structure:
            - created_at (str): Creation date and time
            - expires_at (str): Expiration date and time
            - md_permissions (list[dict]): Market data access information
                - quote_level (str): Quote level
                - delay_minutes (float): Delay in minutes
                - mic (str): Exchange MIC identifier
                - country (str): Country
                - continent (str): Continent
                - worldwide (bool): Worldwide access
            - account_ids (list[str]): Account identifiers
            - readonly (bool): Session and trading accounts marked as readonly
    """
    await _ensure_authorized()
    if token:
        return api_client.execute_request(
            "POST",
            "/v1/sessions/details",
            json={"token": token},
        )
    return api_client.execute_request("GET", "/v1/sessions/details")


@mcp.tool()
async def GetAccount(account_id: str) -> Dict[str, Any]:
    """
    Get information about specific account

    Args:
        account_id: Account identifier

    Returns:
        dict: Account information with the following structure:
            - account_id (str): Account identifier
            - type (str): Account type
            - status (str): Account status
            - equity (str): Available funds plus value of open positions
            - unrealized_profit (str): Unrealized profit
            - positions (list[dict]): Positions (open plus theoretical from active unfilled orders)
                - symbol (str): Instrument symbol
                - quantity (str): Quantity in units, signed value determining (long-short)
                - average_price (str): Average price. Not filled for FORTS positions
                - current_price (str): Current price
                - maintenance_margin (str): Maintenance margin. Filled only for FORTS positions
                - daily_pnl (str): Profit or loss for current day (PnL). Not filled for FORTS positions
                - unrealized_pnl (str): Total unrealized profit or loss (PnL) of current position
            - cash (list[dict]): Own cash available for trading. Does not include margin funds
            - portfolio_mc (dict): General type for Moscow Exchange accounts. Includes both unified and mono accounts
                - available_cash (str): Own cash available for trading. Includes margin funds
                - initial_margin (str): Initial margin
                - maintenance_margin (str): Maintenance margin
            - portfolio_mct (dict): Portfolio type for US market accounts
            - portfolio_forts (dict): Portfolio type for trading on Moscow Exchange futures market
                - available_cash (str): Own cash available for trading. Includes margin funds
                - money_reserved (str): Minimum margin (required collateral for open positions)
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", f"/v1/accounts/{account_id}")


@mcp.tool()
async def Trades(
    account_id: str,
    limit: str = "none",
    interval_start: str = "none",
    interval_end: str = "none",
) -> Dict[str, Any]:
    """
    Get account trade history

    Args:
        account_id: Account identifier
        limit: чтобы узнать последние N сделок по счету (may be str or "none")
        interval_start: Start of requested period, Unix epoch time (may be "none")
        interval_end: End of requested period, Unix epoch time (may be "none)

    Returns:
        dict: Trade history with the following structure:
            - trades (list[dict]): Account trades (AccountTrade objects)
    """
    params: Dict[str, str] = {}
    if limit != "none":
        params["limit"] = str(limit)
    if interval_start != "none":
        params["interval.start_time"] = str(interval_start)
    if interval_end != "none":
        params["interval.end_time"] = str(interval_end)

    await _ensure_authorized()
    return api_client.execute_request(
        "GET",
        f"/v1/accounts/{account_id}/trades",
        params=params or None,
    )


@mcp.tool()
async def Transactions(
    account_id: str,
    limit: str = "none",
    interval_start: str = "none",
    interval_end: str = "none",
) -> Dict[str, Any]:
    """
    Get list of account transactions

    Args:
        account_id: Account identifier
        limit: чтобы узнать последние N транзакций по счету (may be str or "none")
        interval_start: Start of requested period, Unix epoch time (may be "none")
        interval_end: End of requested period, Unix epoch time (may be "none)

    Returns:
        dict: Transactions with the following structure:
            - transactions (list[dict]): Account transactions
                - id (str): Transaction identifier
                - category (str): Transaction type from TransactionCategory
                - timestamp (str): Timestamp
                - symbol (str): Instrument symbol
                - change (dict): Money change (google.type.Money)
                - trade (dict): Trade information
                - transaction_category (str): Transaction category from TransactionCategory
                - transaction_name (str): Transaction name
    """
    params: Dict[str, str] = {}
    if limit != "none":
        params["limit"] = str(limit)
    if interval_start != "none":
        params["interval.start_time"] = str(interval_start)
    if interval_end != "none":
        params["interval.end_time"] = str(interval_end)

    await _ensure_authorized()
    return api_client.execute_request(
        "GET",
        f"/v1/accounts/{account_id}/transactions",
        params=params or None,
    )


@mcp.tool()
async def Clock_ACCOUNTS(account_id: str) -> Dict[str, Any]:
    """
    Get server time (ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ТУЛ Clock_ACCOUNTS ЕСЛИ ТРЕБУЕТСЯ УЗНАТЬ ТРАНЗАКЦИИ ИЛИ СДЕЛКИ ВО ВРЕМЕННОМ ПРОМЕЖУТКЕ, КВАРТАЛЕ И ТД)
    ДЛЯ КВАРТАЛА И ВРЕМЕННОГО ПРОМЕЖТУКА ДЛЯ НАЧАЛА УЗНАЙ ТЕКУЩЕЕ ВРЕМЯ interval_end С ПОМОЩЬЮ ИНСТРУМЕНТА Clock_ACCOUNTS

    Args:
        account_id: Account identifier

    Returns:
        dict: Server time with the following structure:
            - timestamp (str): Timestamp
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", "/v1/assets/clock")


@mcp.tool()
async def Exchanges(account_id: str) -> Dict[str, Any]:
    """
    Get list of available exchanges with names and mic codes

    Args:
        account_id: Account identifier

    Returns:
        dict: List of exchanges with the following structure:
            - exchanges (list[dict]): Exchange information
                - mic (str): Exchange MIC identifier
                - name (str): Exchange name
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", "/v1/exchanges")


@mcp.tool()
async def GetAsset(symbol: str, account_id: str) -> Dict[str, Any]:
    """
    Get information about specific instrument

    Args:
        symbol: Instrument symbol
        account_id

    Returns:
        dict: Instrument information with the following structure:
            - board (str): Trading mode code
            - id (str): Instrument identifier
            - ticker (str): Instrument ticker
            - mic (str): Exchange MIC identifier
            - isin (str): Instrument ISIN identifier
            - type (str): Instrument type
            - name (str): Instrument name
            - decimals (int): Number of decimal places in price
            - min_step (str): Minimum price step. For final price step calculation: min_step/(10^decimals)
            - lot_size (str): Number of units in lot
            - expiration_date (dict): Futures expiration date (google.type.Date)
            - quote_currency (str): Quote currency, may differ from instrument trading mode currency
    """
    params: Optional[Dict[str, str]] = {"account_id": account_id} if account_id else None
    await _ensure_authorized()
    return api_client.execute_request(
        "GET",
        f"/v1/assets/{symbol}",
        params=params,
    )


@mcp.tool()
async def GetAssetParams(symbol: str, account_id: str = "") -> Dict[str, Any]:
    """
    Get trading parameters for instrument

    Args:
        symbol: Instrument symbol
        account_id: (счет) только для проверки информации об акциях на счете

    Returns:
        dict: Trading parameters with the following structure:
            - symbol (str): Instrument symbol
            - account_id (str): Account ID for which trading parameters are selected
            - tradeable (bool): Are trading operations available
            - longable (dict): Are long operations available
                - value (str): Instrument status
                - halted_days (int): How many days long operations are prohibited (if any)
            - shortable (dict): Are short operations available
                - value (str): Instrument status
                - halted_days (int): How many days short operations are prohibited (if any)
            - long_risk_rate (str): Risk rate for long operation
            - long_collateral (dict): Collateral amount to maintain long position (google.type.Money)
            - short_risk_rate (str): Risk rate for short operation
            - short_collateral (dict): Collateral amount to maintain short position (google.type.Money)
            - long_initial_margin (dict): Initial requirements, how much free cash must be in account to open long position, for FORTS accounts equals exchange margin
            - short_initial_margin (dict): Initial requirements, how much free cash must be in account to open short position, for FORTS accounts equals exchange margin
    """
    params: Optional[Dict[str, str]]
    if account_id and ":" not in account_id:
        params = {"account_id": account_id}
    else:
        params = None

    await _ensure_authorized()
    return api_client.execute_request(
        "GET",
        f"/v1/assets/{symbol}/params",
        params=params,
    )


@mcp.tool()
async def OptionsChain(underlying_symbol: str) -> Dict[str, Any]:
    """
    Get options chain for underlying asset

    Args:
        underlying_symbol: Underlying asset symbol for option

    Returns:
        dict: Options chain with the following structure:
            - symbol (str): Underlying asset symbol for option
            - options (list[dict]): Option information
                - symbol (str): Instrument symbol
                - type (str): Instrument type
                - contract_size (str): Lot, quantity of underlying asset in instrument
                - trade_first_day (dict): Trading start date (google.type.Date)
                - trade_last_day (dict): Trading end date (google.type.Date)
                - strike (str): Option strike price
                - multiplier (str): Option multiplier
                - expiration_first_day (dict): Expiration start date (google.type.Date)
                - expiration_last_day (dict): Expiration end date (google.type.Date)
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", f"/v1/assets/{underlying_symbol}/options")


@mcp.tool()
async def Schedule(symbol: str) -> Dict[str, Any]:
    """
    Get trading schedule for instrument

    Args:
        symbol: Instrument symbol

    Returns:
        dict: Trading schedule with the following structure:
            - symbol (str): Instrument symbol
            - sessions (list[dict]): Instrument sessions
                - type (str): Session type
                - interval (dict): Session interval (google.type.Interval)
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", f"/v1/assets/{symbol}/schedule")


@mcp.tool()
async def CancelOrder(account_id: str, order_id: str) -> Dict[str, Any]:
    """
    Cancel exchange order

    Args:
        account_id: Account identifier
        order_id: Order identifier

    Returns:
        dict: Cancelled order information with the following structure:
            - order_id (str): Order identifier
            - exec_id (str): Execution identifier
            - status (str): Order status (OrderStatus)
            - order (dict): Order
                - account_id (str): Account identifier
                - symbol (str): Instrument symbol
                - quantity (str): Quantity in units
                - side (str): Side (long or short)
                - type (str): Order type (OrderType)
                - time_in_force (str): Time in force (TimeInForce)
                - limit_price (str): Required for limit and stop limit orders
                - stop_price (str): Required for stop market and stop limit orders
                - stop_condition (str): Required for stop market and stop limit orders
                - legs (list[dict]): Required for multi-leg orders
                - client_order_id (str): Unique order identifier. Auto-generated if not sent (max 20 characters)
                - valid_before (dict): Conditional order validity period. Filled for ORDER_TYPE_STOP, ORDER_TYPE_STOP_LIMIT orders
                - comment (str): Order label (max 128 characters)
            - transact_at (str): Order placement date and time
            - accept_at (str): Order acceptance date and time
            - withdraw_at (str): Order cancellation date and time
    """
    await _ensure_authorized()
    return api_client.execute_request("DELETE", f"/v1/accounts/{account_id}/orders/{order_id}")


@mcp.tool()
async def GetOrder(account_id: str, order_id: str) -> Dict[str, Any]:
    """
    Get information about specific order

    Args:
        account_id: Account identifier
        order_id: Order identifier

    Returns:
        dict: Order information with the following structure:
            - order_id (str): Order identifier
            - exec_id (str): Execution identifier
            - status (str): Order status (OrderStatus)
            - order (dict): Order
                - account_id (str): Account identifier
                - symbol (str): Instrument symbol
                - quantity (str): Quantity in units
                - side (str): Side (long or short)
                - type (str): Order type (OrderType)
                - time_in_force (str): Time in force (TimeInForce)
                - limit_price (str): Required for limit and stop limit orders
                - stop_price (str): Required for stop market and stop limit orders
                - stop_condition (str): Required for stop market and stop limit orders
                - legs (list[dict]): Required for multi-leg orders
                - client_order_id (str): Unique order identifier. Auto-generated if not sent (max 20 characters)
                - valid_before (dict): Conditional order validity period. Filled for ORDER_TYPE_STOP, ORDER_TYPE_STOP_LIMIT orders
                - comment (str): Order label (max 128 characters)
            - transact_at (str): Order placement date and time
            - accept_at (str): Order acceptance date and time
            - withdraw_at (str): Order cancellation date and time
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", f"/v1/accounts/{account_id}/orders/{order_id}")


@mcp.tool()
async def GetOrders(account_id: str) -> Dict[str, Any]:
    """
    Get list of orders for account

    Args:
        account_id: Account identifier

    Returns:
        dict: List of orders with the following structure:
            - orders (list[dict]): Orders (OrderState objects)
                - order_id (str): Order identifier
                - exec_id (str): Execution identifier
                - status (str): Order status (OrderStatus)
                - order (dict): Order
                - transact_at (str): Order placement date and time
                - accept_at (str): Order acceptance date and time
                - withdraw_at (str): Order cancellation date and time
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", f"/v1/accounts/{account_id}/orders")


@mcp.tool()
async def PlaceOrder(
    account_id: str,
    symbol: str,
    quantity: str,
    side: str,
    type: str,
    time_in_force: str,
    limit_price: Optional[str] = None,
    stop_price: Optional[str] = None,
    stop_condition: Optional[str] = None,
    legs: Optional[List[Dict[str, Any]]] = None,
    client_order_id: Optional[str] = None,
    valid_before: Optional[Dict[str, Any]] = None,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Place exchange order

    Args:
        account_id: Account identifier
        symbol: Instrument symbol
        quantity: Quantity in units
        side: Side (long or short)
        type: Order type (OrderType)
        time_in_force: Time in force (TimeInForce)
        limit_price: Required for limit and stop limit orders (optional)
        stop_price: Required for stop market and stop limit orders (optional)
        stop_condition: Required for stop market and stop limit orders (optional)
        legs: Required for multi-leg orders (optional)
        client_order_id: Unique order identifier. Auto-generated if not sent (max 20 characters) (optional)
        valid_before: Conditional order validity period. Filled for ORDER_TYPE_STOP, ORDER_TYPE_STOP_LIMIT orders (optional)
        comment: Order label (max 128 characters) (optional)

    Returns:
        dict: Placed order information with the following structure:
            - order_id (str): Order identifier
            - exec_id (str): Execution identifier
            - status (str): Order status (OrderStatus)
            - order (dict): Order
                - account_id (str): Account identifier
                - symbol (str): Instrument symbol
                - quantity (str): Quantity in units
                - side (str): Side (long or short)
                - type (str): Order type (OrderType)
                - time_in_force (str): Time in force (TimeInForce)
                - limit_price (str): Required for limit and stop limit orders
                - stop_price (str): Required for stop market and stop limit orders
                - stop_condition (str): Required for stop market and stop limit orders
                - legs (list[dict]): Required for multi-leg orders
                - client_order_id (str): Unique order identifier. Auto-generated if not sent (max 20 characters)
                - valid_before (dict): Conditional order validity period. Filled for ORDER_TYPE_STOP, ORDER_TYPE_STOP_LIMIT orders
                - comment (str): Order label (max 128 characters)
            - transact_at (str): Order placement date and time
            - accept_at (str): Order acceptance date and time
            - withdraw_at (str): Order cancellation date and time
    """
    data: Dict[str, Any] = {
        "symbol": symbol,
        "quantity": quantity,
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
    }
    if limit_price is not None:
        data["limit_price"] = limit_price
    if stop_price is not None:
        data["stop_price"] = stop_price
    if stop_condition is not None:
        data["stop_condition"] = stop_condition
    if legs is not None:
        data["legs"] = legs
    if client_order_id is not None:
        data["client_order_id"] = client_order_id
    if valid_before is not None:
        data["valid_before"] = valid_before
    if comment is not None:
        data["comment"] = comment

    await _ensure_authorized()
    return api_client.execute_request(
        "POST",
        f"/v1/accounts/{account_id}/orders",
        json=data,
    )


@mcp.tool()
async def Clock_MARKET_DATA(account_id: str) -> Dict[str, Any]:
    """
    Get server time (ALWAYS USE THIS TOOL IF YOU NEED TO OBTAIN DATA FOR A TIME PERIOD OR QUARTER)
    FIRST, FETCH interval_end AND HISTORICAL DATA WITH TOOL Clock_MARKET_DATA IF YOU NEED TO RETRIEVE PRICE HISTORY FOR THE LAST QUARTER OR A SPECIFIC TIME PERIOD.
    NEVER SET interval_end MANUALLY.

    Args:
        account_id: Account identifier

    Returns:
        dict: Server time with the following structure:
            - timestamp (str): Timestamp
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", "/v1/assets/clock")


@mcp.tool()
async def Bars(
    symbol: str,
    timeframe: str,
    interval_start: str = "none",
    interval_end: str = "none",
) -> Dict[str, Any]:
    """
    Get data for instrument (aggregated candles)

    Args:
        symbol: Instrument symbol
        timeframe: Required timeframe (may be "none")
        interval_start: Start of requested period (may be "none")
        interval_end: End of requested period (may be "none")

    Returns:
        dict: Historical data with the following structure:
            - symbol (str): Instrument symbol
            - bars (list[dict]): Aggregated candle
                - timestamp (str): Timestamp
                - open (str): Candle open price
                - high (str): Candle high price
                - low (str): Candle low price
                - close (str): Candle close price
                - volume (str): Trading volume for candle in units
    """
    params: Dict[str, str] = {"timeframe": timeframe}
    if interval_start != "none":
        params["interval.start_time"] = str(interval_start)
    if interval_end != "none":
        params["interval.end_time"] = str(interval_end)

    await _ensure_authorized()
    return api_client.execute_request(
        "GET",
        f"/v1/instruments/{symbol}/bars",
        params=params,
    )


@mcp.tool()
async def LastQuote(symbol: str) -> Dict[str, Any]:
    """
    Get latest quote for instrument

    Args:
        symbol: Instrument symbol

    Returns:
        dict: Latest quote with the following structure:
            - symbol (str): Instrument symbol
            - quote (dict): Last trade price
                - symbol (str): Instrument symbol
                - timestamp (str): Timestamp
                - ask (str): Ask. 0 when no active ask
                - ask_size (str): Ask size
                - bid (str): Bid. 0 when no active bid
                - bid_size (str): Bid size
                - last (str): Last trade price
                - last_size (str): Last trade size
                - volume (str): Daily trade volume
                - turnover (str): Daily trade turnover
                - open (str): Open price. Daily
                - high (str): High price. Daily
                - low (str): Low price. Daily
                - close (str): Close price. Daily
                - change (str): Price change (last minus close)
                - option (dict): Option information
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", f"/v1/instruments/{symbol}/quotes/latest")


@mcp.tool()
async def LatestTrades(symbol: str) -> Dict[str, Any]:
    """
    Get list of latest trades for instrument

    Args:
        symbol: Instrument symbol

    Returns:
        dict: Latest trades with the following structure:
            - symbol (str): Instrument symbol
            - trades (list[dict]): List of latest trades
                - trade_id (str): Trade identifier sent by exchange
                - mpid (str): Market participant identifier
                - timestamp (str): Timestamp
                - price (str): Trade price
                - size (str): Trade size
                - side (str): Trade side (buy or sell)
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", f"/v1/instruments/{symbol}/trades/latest")


@mcp.tool()
async def OrderBook(symbol: str) -> Dict[str, Any]:
    """
    Get current order book for instrument

    Args:
        symbol: Instrument symbol

    Returns:
        dict: Order book with the following structure:
            - symbol (str): Instrument symbol
            - orderbook (dict): Order book
                - rows (list[dict]): Order book levels (OrderBook.Row)
    """
    await _ensure_authorized()
    return api_client.execute_request("GET", f"/v1/instruments/{symbol}/orderbook")


if __name__ == "__main__":
    mcp.run()
