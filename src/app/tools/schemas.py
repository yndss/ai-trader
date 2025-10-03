import datetime as dt
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Account(BaseModel):
    id: str
    name: str
    currency: str
    balance: float
    buying_power: float

class Position(BaseModel):
    figi: str
    ticker: str
    qty: float
    avg_price: float
    market_price: Optional[float] = None
    pnl_abs: Optional[float] = None
    pnl_pct: Optional[float] = None
    sector: Optional[str] = None

class Candle(BaseModel):
    time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class FindInstrumentInput(BaseModel):
    query: str

class GetCandlesInput(BaseModel):
    figi: str
    timeframe: Literal["1m","5m","15m","1h","1d"] = "1d"
    start: dt.datetime
    end: dt.datetime

class GetOrderbookInput(BaseModel):
    figi: str
    depth: int = Field(ge=1, le=50, default=10)

class PlaceOrderInput(BaseModel):
    account_id: str
    figi: str
    side: Literal["buy","sell"]
    qty: float
    type: Literal["market","limit"] = "market"
    limit_price: Optional[float] = None

class CancelOrderInput(BaseModel):
    account_id: str
    order_id: str

class GetPortfolioInput(BaseModel):
    account_id: str
class GetPortfolioInput(BaseModel):
    account_id: str
