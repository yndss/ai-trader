from __future__ import annotations

"""FastMCP server exposing Finam TradeAPI endpoints as tools."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

try:
    from src.finam_client import FinamAPIClient
except ImportError:  # pragma: no cover - fallback if namespace import fails
    from src.finam_client.client import FinamAPIClient  # type: ignore

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

_DEFAULT_SECRET = os.getenv("FINAM_AUTH_SECRET") or os.getenv("FINAM_ACCESS_TOKEN") or ""
_CURRENT_TOKEN: Optional[str] = None

mcp = FastMCP("FinamTrader")
api_client = FinamAPIClient(
    access_token=_DEFAULT_SECRET
)


def _set_authorization(token: Optional[str]) -> None:
    global _CURRENT_TOKEN
    if token:
        formatted = token.strip()
        api_client.access_token = formatted  # Обновляем токен в самом клиенте
        api_client.session.headers["Authorization"] = formatted
        _CURRENT_TOKEN = formatted
    else:
        api_client.access_token = ""  # Очищаем токен в самом клиенте
        api_client.session.headers.pop("Authorization", None)
        _CURRENT_TOKEN = None


_initial_auth = api_client.session.headers.get("Authorization")
if _initial_auth:
    _set_authorization(_initial_auth)
elif _DEFAULT_SECRET:
    # Убеждаемся, что токен из переменной окружения установлен в заголовки
    _set_authorization(_DEFAULT_SECRET)


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


# ==================== AUTH ====================

@mcp.tool()
async def Auth(secret = "") -> dict:
    """
    Get JWT token from API token
    
    Args:
        secret: API token (secret key)
    
    Returns:
        dict: JWT token information with the following structure:
            - token (str): Received JWT token
    """
    return api_client.execute_request("POST", "/v1/sessions", json={"secret": secret})

@mcp.tool()
async def TokenDetails(token = "") -> dict:
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
    return api_client.execute_request("POST", "/v1/sessions/details", json={"token": token})

# ==================== ACCOUNTS ====================

@mcp.tool()
async def GetAccount(account_id = "") -> dict:
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
    request = api_client.execute_request("GET", f"/v1/accounts/{account_id}")

    return request


@mcp.tool()
async def Trades(account_id = "", limit: str = "none", interval_start: str = "none", interval_end: str = "none") -> dict:
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

    if limit != "none":
        return api_client.execute_request("GET", f"/v1/accounts/{account_id}/trades/limit={limit}") 
    if interval_start != "none" and interval_end != "none":
        return api_client.execute_request("GET", f"/v1/accounts/{account_id}/trades?interval.start_time={interval_start}&interval.end_time={interval_end}")
    return api_client.execute_request("GET", f"/v1/accounts/{account_id}/trades")


@mcp.tool()
async def Transactions(account_id = "", limit: str = "none", interval_start: str = "none", interval_end: str = "none") -> dict:
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
    if limit != "none":
        return api_client.execute_request("GET", f"/v1/accounts/{account_id}/transactions/limit={limit}") 
    if interval_start != "none" and interval_end != "none":
        return api_client.execute_request("GET", f"/v1/accounts/{account_id}/transactions?interval.start_time={interval_start}&interval.end_time={interval_end}")
    return api_client.execute_request("GET", f"/v1/accounts/{account_id}/transactions")

@mcp.tool()
async def Clock_ACCOUNTS(account_id = "") -> dict:
    """
    Get server time (ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ТУЛ Clock_ACCOUNTS ЕСЛИ ТРЕБУЕТСЯ УЗНАТЬ ТРАНЗАКЦИИ ИЛИ СДЕЛКИ ВО ВРЕМЕННОМ ПРОМЕЖУТКЕ, КВАРТАЛЕ И ТД)
    ДЛЯ КВАРТАЛА И ВРЕМЕННОГО ПРОМЕЖТУКА ДЛЯ НАЧАЛА УЗНАЙ ТЕКУЩЕЕ ВРЕМЯ interval_end С ПОМОЩЬЮ ИНСТРУМЕНТА Clock_ACCOUNTS
    
    Args:
        account_id: Account identifier
    
    Returns:
        dict: Server time with the following structure:
            - timestamp (str): Timestamp
    """
    return api_client.execute_request("GET", "/v1/assets/clock")

# ==================== INSTRUMENTS ====================

@mcp.tool()
async def Clock(account_id = "") -> dict:
    """
    Get server time
    
    Args:
        account_id: Account identifier
    
    Returns:
        dict: Server time with the following structure:
            - timestamp (str): Timestamp
    """
    return api_client.execute_request("GET", "/v1/assets/clock")

@mcp.tool()
async def Assets(account_id = "") -> dict:
    """
    Get list of available instruments and their descriptions
    
    Args:
        account_id: Account identifier
    
    Returns:
        dict: Instruments list with the following structure:
            - assets (list): Instrument information
                - symbol (str): Instrument symbol ticker@mic
                - id (str): Instrument identifier
                - ticker (str): Instrument ticker
                - mic (str): Exchange MIC identifier
                - isin (str): Instrument ISIN identifier
                - type (str): Instrument type
                - name (str): Instrument name
    """
    return api_client.execute_request("GET", "/v1/assets")


@mcp.tool()
async def Exchanges(account_id = "") -> dict:
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
    return api_client.execute_request("GET", "/v1/exchanges")


@mcp.tool()
async def GetAsset(symbol = "", account_id = "") -> dict:
    """
    Get information about specific instrument
    
    Args:
        symbol: Instrument symbol
        account_id: (счет) только для проверки информации об акциях на счете 
    
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

    if account_id != "":
        return api_client.execute_request("GET", f"/v1/assets/{symbol}?account_id={account_id}")
    return api_client.execute_request("GET", f"/v1/assets/{symbol}")


@mcp.tool()
async def GetAssetParams(symbol = "", account_id: str = "") -> dict:
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

    if ":" not in account_id:
        return api_client.execute_request("GET", f"/v1/assets/{symbol}/params?account_id={account_id}")
    return api_client.execute_request("GET", f"/v1/assets/{symbol}/params")


@mcp.tool()
async def OptionsChain(underlying_symbol = "") -> dict:
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

    return api_client.execute_request("GET", f"/v1/assets/{underlying_symbol}/options")


@mcp.tool()
async def Schedule(symbol = "") -> dict:
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
    return api_client.execute_request("GET", f"/v1/assets/{symbol}/schedule")

# ==================== ORDERS ====================

@mcp.tool()
async def CancelOrder(account_id = "", order_id = "") -> dict:
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
    return api_client.execute_request("DELETE", f"/v1/accounts/{account_id}/orders/{order_id}")


@mcp.tool()
async def GetOrder(account_id = "", order_id = "") -> dict:
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
    return api_client.execute_request("GET", f"/v1/accounts/{account_id}/orders/{order_id}")


@mcp.tool()
async def GetOrders(account_id = "") -> dict:
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
    return api_client.execute_request("GET", f"/v1/accounts/{account_id}/orders")


@mcp.tool()
async def PlaceOrder(
    account_id: str,
    symbol: str,
    quantity: str,
    side: str,
    type: str,
    time_in_force: str,
    limit_price: str = None,
    stop_price: str = None,
    stop_condition: str = None,
    legs: list = None,
    client_order_id: str = None,
    valid_before: dict = None,
    comment: str = None
) -> dict:
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
    data = {
        "symbol": symbol,
        "quantity": quantity,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
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
    
    return api_client.execute_request("POST", f"/v1/accounts/{account_id}/orders", json=data)

# ==================== MARKET_DATA ====================

@mcp.tool()
async def Clock_MARKET_DATA(account_id = "") -> dict:
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
    return api_client.execute_request("GET", "/v1/assets/clock")

@mcp.tool()
async def Bars(
    symbol: str,
    timeframe: str,
    interval_start: str = "none",
    interval_end: str = "none"
) -> dict:
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
    params = {"timeframe": timeframe}
    if interval_start != "none":
        params["interval_start"] = interval_start
    if interval_end != "none":
        params["interval_end"] = interval_end
        return api_client.execute_request("GET", f"/v1/instruments/{symbol}/bars?timeframe={timeframe}&interval.start_time={interval_start}&interval.end_time={interval_end}") 
    return api_client.execute_request("GET", f"/v1/instruments/{symbol}/bars")


@mcp.tool()
async def LastQuote(symbol: str) -> dict:
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
    return api_client.execute_request("GET", f"/v1/instruments/{symbol}/quotes/latest")


@mcp.tool()
async def LatestTrades(symbol: str) -> dict:
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
    return api_client.execute_request("GET", f"/v1/instruments/{symbol}/trades/latest")


@mcp.tool()
async def OrderBook(symbol: str) -> dict:
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
    return api_client.execute_request("GET", f"/v1/instruments/{symbol}/orderbook")


if __name__ == "__main__":
    mcp.run()


if __name__ == "__main__":
    mcp.run()
