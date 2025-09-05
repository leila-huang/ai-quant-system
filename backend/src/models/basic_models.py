"""
基础数据模型 - 不使用Pydantic，避免版本兼容性问题
"""

from datetime import date
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class StockDailyBar:
    """
    股票日线数据模型
    """
    date: date
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: float
    amount: Optional[float] = None
    adjust_factor: Optional[float] = None


@dataclass
class StockData:
    """
    单只股票的所有历史数据
    """
    symbol: str
    name: Optional[str] = None
    bars: Optional[List[StockDailyBar]] = None
    
    def __post_init__(self):
        if self.bars is None:
            self.bars = []
