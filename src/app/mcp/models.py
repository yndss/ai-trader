from __future__ import annotations

"""Pydantic models and enums for Finam TradeAPI MCP integration."""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# -------------------- ENUMS --------------------


class Side(str, Enum):
    """Order side as expected by the REST API."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported order types for placement requests."""

    ORDER_TYPE_UNSPECIFIED = "ORDER_TYPE_UNSPECIFIED"
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MULTI_LEG = "MULTI_LEG"


class TimeInForce(str, Enum):
    """Time-in-force policies recognised by the REST API."""

    TIME_IN_FORCE_UNSPECIFIED = "TIME_IN_FORCE_UNSPECIFIED"
    DAY = "DAY"
    GTC = "GOOD_TILL_CANCEL"
    GTX = "GOOD_TILL_CROSSING"
    EXT = "EXT"
    ON_OPEN = "ON_OPEN"
    ON_CLOSE = "ON_CLOSE"
    IOC = "IOC"
    FOK = "FOK"


class StopCondition(str, Enum):
    """Supported stop trigger conditions."""

    STOP_CONDITION_UNSPECIFIED = "STOP_CONDITION_UNSPECIFIED"
    LAST_UP = "LAST_UP"
    LAST_DOWN = "LAST_DOWN"


class ValidBefore(str, Enum):
    """Order good-till validator."""

    VALID_BEFORE_UNSPECIFIED = "VALID_BEFORE_UNSPECIFIED"
    END_OF_DAY = "END_OF_DAY"
    GOOD_TILL_CANCEL = "GOOD_TILL_CANCEL"
    GOOD_TILL_DATE = "GOOD_TILL_DATE"


class OrderStatus(str, Enum):
    """Possible states returned for orders."""

    ORDER_STATUS_UNSPECIFIED = "ORDER_STATUS_UNSPECIFIED"
    NEW = "ORDER_STATUS_NEW"
    PARTIALLY_FILLED = "ORDER_STATUS_PARTIALLY_FILLED"
    FILLED = "ORDER_STATUS_FILLED"
    DONE_FOR_DAY = "ORDER_STATUS_DONE_FOR_DAY"
    CANCELED = "ORDER_STATUS_CANCELED"
    REPLACED = "ORDER_STATUS_REPLACED"
    PENDING_CANCEL = "ORDER_STATUS_PENDING_CANCEL"
    REJECTED = "ORDER_STATUS_REJECTED"
    SUSPENDED = "ORDER_STATUS_SUSPENDED"
    PENDING_NEW = "ORDER_STATUS_PENDING_NEW"
    EXPIRED = "ORDER_STATUS_EXPIRED"
    FAILED = "ORDER_STATUS_FAILED"
    FORWARDING = "ORDER_STATUS_FORWARDING"
    WAIT = "ORDER_STATUS_WAIT"
    DENIED_BY_BROKER = "ORDER_STATUS_DENIED_BY_BROKER"
    REJECTED_BY_EXCHANGE = "ORDER_STATUS_REJECTED_BY_EXCHANGE"
    WATCHING = "ORDER_STATUS_WATCHING"
    EXECUTED = "ORDER_STATUS_EXECUTED"
    DISABLED = "ORDER_STATUS_DISABLED"
    LINK_WAIT = "ORDER_STATUS_LINK_WAIT"
    SL_GUARD_TIME = "ORDER_STATUS_SL_GUARD_TIME"
    SL_EXECUTED = "ORDER_STATUS_SL_EXECUTED"
    SL_FORWARDING = "ORDER_STATUS_SL_FORWARDING"
    TP_GUARD_TIME = "ORDER_STATUS_TP_GUARD_TIME"
    TP_EXECUTED = "ORDER_STATUS_TP_EXECUTED"
    TP_CORRECTION = "ORDER_STATUS_TP_CORRECTION"
    TP_FORWARDING = "ORDER_STATUS_TP_FORWARDING"
    TP_CORR_GUARD_TIME = "ORDER_STATUS_TP_CORR_GUARD_TIME"


class TimeFrame(str, Enum):
    """Market data aggregation intervals."""

    TIME_FRAME_UNSPECIFIED = "TIME_FRAME_UNSPECIFIED"
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H2 = "H2"
    H4 = "H4"
    H8 = "H8"
    D = "D"
    W = "W"
    MN = "MN"
    QR = "QR"


class QuoteLevel(str, Enum):
    """Market data permission levels."""

    QUOTE_LEVEL_UNSPECIFIED = "QUOTE_LEVEL_UNSPECIFIED"
    LAST_PRICE = "QUOTE_LEVEL_LAST_PRICE"
    BEST_BID_OFFER = "QUOTE_LEVEL_BEST_BID_OFFER"
    DEPTH_OF_MARKET = "QUOTE_LEVEL_DEPTH_OF_MARKET"
    DEPTH_OF_BOOK = "QUOTE_LEVEL_DEPTH_OF_BOOK"
    ACCESS_FORBIDDEN = "QUOTE_LEVEL_ACCESS_FORBIDDEN"


class OrderBookRowAction(str, Enum):
    """Possible actions for order book row deltas."""

    ACTION_UNSPECIFIED = "ACTION_UNSPECIFIED"
    REMOVE = "ACTION_REMOVE"
    ADD = "ACTION_ADD"
    UPDATE = "ACTION_UPDATE"


# -------------------- AUTH --------------------


class AuthRequest(BaseModel):
    """Payload for POST /v1/sessions."""

    secret: str = Field(..., description="API secret")


class AuthResponse(BaseModel):
    """Response with issued JWT token."""

    token: str = Field(..., description="JWT token")


class TokenDetailsRequest(BaseModel):
    """Payload for POST /v1/sessions/details."""

    token: str = Field(..., description="JWT token")


class MDPermission(BaseModel):
    quote_level: QuoteLevel
    delay_minutes: int
    mic: str
    country: str
    continent: str
    worldwide: bool


class TokenDetailsResponse(BaseModel):
    created_at: datetime
    expires_at: datetime
    md_permissions: List[MDPermission]
    account_ids: List[str]
    readonly: Optional[bool] = None


# -------------------- ORDERS --------------------


def _decimalish_to_str(
    value: str | int | float | Decimal | None,
    *,
    positive: bool = True,
) -> Optional[str]:
    if value is None:
        return None
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("must be a decimal-like value") from exc
    if positive and decimal_value <= 0:
        raise ValueError("must be > 0")
    return format(decimal_value, "f")


class Leg(BaseModel):
    symbol: str = Field(..., description="ticker@mic")
    quantity: str = Field(..., description="Quantity as string")
    side: Side

    @field_validator("symbol")
    @classmethod
    def _symbol_has_mic(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Symbol must be in format TICKER@MIC (e.g., SBER@TQBR)")
        return value

    @field_validator("quantity", mode="before")
    @classmethod
    def _quantity_to_str(cls, value: Any) -> str:
        string_value = _decimalish_to_str(value, positive=True)
        assert string_value is not None
        return string_value

    def to_request_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side.value,
        }


class Order(BaseModel):
    account_id: str
    symbol: str
    quantity: str = Field(..., description="Quantity as string")
    side: Side
    type: OrderType
    time_in_force: Optional[TimeInForce] = Field(None, description="Required for MARKET/LIMIT")
    limit_price: Optional[str] = Field(None, description="String price for LIMIT")
    stop_price: Optional[str] = Field(None, description="String stop trigger")
    stop_condition: Optional[StopCondition] = None
    legs: Optional[List[Leg]] = None
    client_order_id: Optional[str] = Field(None, max_length=20)
    valid_before: Optional[ValidBefore] = None
    comment: Optional[str] = Field(None, max_length=128)

    @field_validator("symbol")
    @classmethod
    def _symbol_has_mic(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Symbol must be in format TICKER@MIC (e.g., AFLT@MISX)")
        return value

    @field_validator("quantity", mode="before")
    @classmethod
    def _quantity_to_str(cls, value: Any) -> str:
        string_value = _decimalish_to_str(value, positive=True)
        assert string_value is not None
        return string_value

    @field_validator("limit_price", mode="before")
    @classmethod
    def _limit_price_to_str(cls, value: Any) -> Optional[str]:
        return _decimalish_to_str(value, positive=True)

    @field_validator("stop_price", mode="before")
    @classmethod
    def _stop_price_to_str(cls, value: Any) -> Optional[str]:
        return _decimalish_to_str(value, positive=True)

    @model_validator(mode="after")
    def _cross_checks(self) -> "Order":
        if self.type == OrderType.LIMIT:
            if not self.limit_price:
                raise ValueError("limit_price is required for LIMIT orders")
            if not self.time_in_force or self.time_in_force == TimeInForce.TIME_IN_FORCE_UNSPECIFIED:
                raise ValueError("time_in_force is required for LIMIT orders")
        if self.type == OrderType.MARKET:
            if self.limit_price or self.stop_price or self.stop_condition:
                raise ValueError("MARKET orders must not define limit/stop fields")
            if not self.time_in_force or self.time_in_force == TimeInForce.TIME_IN_FORCE_UNSPECIFIED:
                raise ValueError("time_in_force is required for MARKET orders")
        if self.type == OrderType.STOP:
            if not self.stop_price or not self.stop_condition:
                raise ValueError("stop_price and stop_condition are required for STOP orders")
            if self.time_in_force is not None and self.time_in_force != TimeInForce.TIME_IN_FORCE_UNSPECIFIED:
                raise ValueError("time_in_force must be omitted for STOP orders")
            if not self.valid_before or self.valid_before == ValidBefore.VALID_BEFORE_UNSPECIFIED:
                raise ValueError("valid_before is required for STOP orders")
        if self.type == OrderType.STOP_LIMIT:
            if not self.limit_price or not self.stop_price or not self.stop_condition:
                raise ValueError("limit_price, stop_price and stop_condition are required for STOP_LIMIT orders")
            if self.time_in_force is not None and self.time_in_force != TimeInForce.TIME_IN_FORCE_UNSPECIFIED:
                raise ValueError("time_in_force must be omitted for STOP_LIMIT orders")
            if not self.valid_before or self.valid_before == ValidBefore.VALID_BEFORE_UNSPECIFIED:
                raise ValueError("valid_before is required for STOP_LIMIT orders")
        if self.client_order_id and len(self.client_order_id) > 20:
            raise ValueError("client_order_id max length is 20")
        if self.comment and len(self.comment) > 128:
            raise ValueError("comment max length is 128")
        return self

    def to_request_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side.value,
            "type": self.type.value,
        }
        if self.time_in_force and self.time_in_force != TimeInForce.TIME_IN_FORCE_UNSPECIFIED:
            payload["time_in_force"] = self.time_in_force.value
        if self.limit_price is not None:
            payload["limit_price"] = self.limit_price
        if self.stop_price is not None:
            payload["stop_price"] = self.stop_price
        if self.stop_condition and self.stop_condition != StopCondition.STOP_CONDITION_UNSPECIFIED:
            payload["stop_condition"] = self.stop_condition.value
        if self.valid_before and self.valid_before != ValidBefore.VALID_BEFORE_UNSPECIFIED:
            payload["valid_before"] = self.valid_before.value
        if self.client_order_id:
            payload["client_order_id"] = self.client_order_id
        if self.comment:
            payload["comment"] = self.comment
        if self.legs:
            payload["legs"] = [leg.to_request_payload() for leg in self.legs]
        return payload


class CancelOrderRequest(BaseModel):
    account_id: str
    order_id: str


class GetOrderRequest(BaseModel):
    account_id: str
    order_id: str


class OrdersRequest(BaseModel):
    account_id: str


class OrderState(BaseModel):
    order_id: str
    exec_id: Optional[str] = None
    status: OrderStatus
    order: Order
    transact_at: Optional[datetime] = None
    accept_at: Optional[datetime] = None
    withdraw_at: Optional[datetime] = None


# -------------------- MARKET DATA --------------------


class Interval(BaseModel):
    """Interval expressed as RFC3339 timestamps."""

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _check_order(self) -> "Interval":
        if self.end <= self.start:
            raise ValueError("interval.end must be greater than interval.start")
        return self


class Bar(BaseModel):
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class BarsRequest(BaseModel):
    symbol: str
    timeframe: TimeFrame
    interval: Interval

    @field_validator("symbol")
    @classmethod
    def _symbol_has_mic(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Symbol must be in format TICKER@MIC (e.g., YDEX@MISX)")
        return value


class BarsResponse(BaseModel):
    symbol: str
    bars: List[Bar]


class QuoteOption(BaseModel):
    open_interest: Optional[Decimal] = None
    implied_volatility: Optional[Decimal] = None
    theoretical_price: Optional[Decimal] = None
    delta: Optional[Decimal] = None
    gamma: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    vega: Optional[Decimal] = None
    rho: Optional[Decimal] = None


class Quote(BaseModel):
    symbol: str
    timestamp: datetime
    ask: Decimal
    ask_size: Decimal
    bid: Decimal
    bid_size: Decimal
    last: Decimal
    last_size: Decimal
    volume: Decimal
    turnover: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    change: Decimal
    option: Optional[QuoteOption] = None


class QuoteRequest(BaseModel):
    symbol: str

    @field_validator("symbol")
    @classmethod
    def _symbol_has_mic(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Symbol must be in format TICKER@MIC")
        return value


class QuoteResponse(BaseModel):
    symbol: str
    quote: Quote


class Trade(BaseModel):
    trade_id: str
    mpid: Optional[str] = ""
    timestamp: datetime
    price: Decimal
    size: Decimal
    side: Side


class LatestTradesRequest(BaseModel):
    symbol: str

    @field_validator("symbol")
    @classmethod
    def _symbol_has_mic(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Symbol must be in format TICKER@MIC")
        return value


class LatestTradesResponse(BaseModel):
    symbol: str
    trades: List[Trade]


class OrderBookRow(BaseModel):
    price: Decimal
    sell_size: Optional[Decimal] = None
    buy_size: Optional[Decimal] = None
    action: OrderBookRowAction
    mpid: Optional[str] = None
    timestamp: datetime


class OrderBook(BaseModel):
    rows: List[OrderBookRow]


class OrderBookRequest(BaseModel):
    symbol: str

    @field_validator("symbol")
    @classmethod
    def _symbol_has_mic(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Symbol must be in format TICKER@MIC")
        return value


class OrderBookResponse(BaseModel):
    symbol: str
    orderbook: OrderBook
