"""
简单数据模型 - 用于Parquet存储引擎
"""

from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field


class StockDailyBar(BaseModel):
    """
    股票日线数据模型
    """
    date: date = Field(..., description="交易日期")
    open_price: float = Field(..., description="开盘价")
    close_price: float = Field(..., description="收盘价")
    high_price: float = Field(..., description="最高价")
    low_price: float = Field(..., description="最低价")
    volume: float = Field(..., description="成交量")
    amount: Optional[float] = Field(None, description="成交额")
    adjust_factor: Optional[float] = Field(None, description="复权因子")


class StockData(BaseModel):
    """
    单只股票的所有历史数据
    """
    symbol: str = Field(..., description="股票代码")
    name: Optional[str] = Field(None, description="股票名称")
    bars: List[StockDailyBar] = Field(..., description="日线数据列表")
