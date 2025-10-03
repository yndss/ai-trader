import datetime as dt
from typing import Any, Dict, List

from ..tools.schemas import Account, Candle, Position


class FinamClient:
    """
    Заглушка/обёртка для Finam TradeAPI. Подключите реальный SDK/REST тут.
    """
    def __init__(self) -> None:
        # TODO: авторизация/создание SDK клиента
        pass

    def get_accounts(self) -> List[Account]:
        return [Account(id="ACC1", name="Primary", currency="RUB", balance=1_000_000.0, buying_power=850_000.0)]

    def get_portfolio(self, account_id: str) -> List[Position]:
        return [
            Position(figi="FIGI_SBER", ticker="SBER", qty=100, avg_price=250.0, market_price=260.0, sector="Financials"),
            Position(figi="FIGI_YNDX", ticker="YNDX", qty=10, avg_price=4000.0, market_price=4200.0, sector="Tech"),
        ]

    def find_instrument(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        if q in {"sber","сбер"}:
            return {"figi":"FIGI_SBER","ticker":"SBER","name":"Sberbank","exchange":"MOEX"}
        if q in {"yndx","яндекс"}:
            return {"figi":"FIGI_YNDX","ticker":"YNDX","name":"Yandex","exchange":"MOEX"}
        return {}

    def get_candles(self, figi: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> List[Candle]:
        import math
        import random
        days = max(5, (end - start).days or 30)
        base = 200.0 if "SBER" in figi else 4000.0
        out: List[Candle] = []
        t = start
        price = base
        for i in range(days):
            price *= 1.0 + 0.002*math.sin(i/5.0) + random.uniform(-0.003,0.003)
            o = price*(1+random.uniform(-0.002,0.002))
            c = price*(1+random.uniform(-0.002,0.002))
            h = max(o,c)*(1+random.uniform(0.0,0.004))
            l = min(o,c)*(1-random.uniform(0.0,0.004))
            v = random.randint(1_000_000, 5_000_000)
            out.append(Candle(time=t, open=o, high=h, low=l, close=c, volume=v))
            t += dt.timedelta(days=1)
        return out

    def get_orderbook(self, figi: str, depth: int) -> Dict[str, Any]:
        return {"bids":[[259.9,1000],[259.8,800]], "asks":[[260.1,1200],[260.2,900]]}

    def place_order(self, account_id: str, figi: str, side: str, qty: float, type_: str, limit_price: float | None):
        return {"order_id":"ORD123","status":"accepted","ts":dt.datetime.utcnow().isoformat()}

    def cancel_order(self, account_id: str, order_id: str):
        return {"order_id":order_id,"status":"canceled"}

    def cancel_order(self, account_id: str, order_id: str):
        return {"order_id":order_id,"status":"canceled"}
