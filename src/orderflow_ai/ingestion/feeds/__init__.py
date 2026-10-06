"""Market data feeds.

Every feed implements the same tiny asynchronous interface, which lets the
pipeline swap a live websocket for a recorded file or a synthetic generator
without any other layer noticing.
"""

from __future__ import annotations

from .base import DepthFeed, Feed, TradeFeed
from .binance import BinanceDepthFeed, BinanceTradeFeed
from .ohlcv import OHLCVFeed
from .replay import ReplayFeed
from .synthetic import SyntheticFeed

__all__ = [
    "BinanceDepthFeed",
    "BinanceTradeFeed",
    "DepthFeed",
    "Feed",
    "OHLCVFeed",
    "ReplayFeed",
    "SyntheticFeed",
    "TradeFeed",
]