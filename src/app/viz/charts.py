import base64
from typing import List

import pandas as pd
import plotly.graph_objects as go

from ..tools.schemas import Candle


def candles_to_base64_png(candles: List[Candle]) -> str:
    df = pd.DataFrame([c.model_dump() for c in candles])
    fig = go.Figure(data=[
        go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])
    ])
    fig.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=400)
    png = fig.to_image(format="png", scale=2)
    return base64.b64encode(png).decode()
    fig.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=400)
    png = fig.to_image(format="png", scale=2)
    return base64.b64encode(png).decode()
