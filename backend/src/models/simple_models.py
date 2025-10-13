"""简单数据模型 - 用于Parquet存储引擎"""

from __future__ import annotations

from datetime import date as Date
from typing import List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class StockDailyBar(BaseModel):
    """股票日线数据模型"""

    model_config = ConfigDict(populate_by_name=True)

    date: Date = Field(
        ...,
        description="交易日期",
        validation_alias=AliasChoices("date", "trade_date"),
    )
    open_price: float = Field(
        ...,
        description="开盘价",
        validation_alias=AliasChoices("open", "open_price"),
    )
    close_price: float = Field(
        ...,
        description="收盘价",
        validation_alias=AliasChoices("close", "close_price"),
    )
    high_price: float = Field(
        ...,
        description="最高价",
        validation_alias=AliasChoices("high", "high_price"),
    )
    low_price: float = Field(
        ...,
        description="最低价",
        validation_alias=AliasChoices("low", "low_price"),
    )
    volume: float = Field(
        ...,
        description="成交量",
        validation_alias=AliasChoices("volume", "vol"),
    )
    amount: Optional[float] = Field(
        None,
        description="成交额",
        validation_alias=AliasChoices("amount", "turnover"),
    )
    adjust_factor: Optional[float] = Field(
        None,
        description="复权因子",
        validation_alias=AliasChoices("adjust_factor", "adj_factor"),
    )

    @property
    def open(self) -> float:
        return self.open_price

    @property
    def close(self) -> float:
        return self.close_price

    @property
    def high(self) -> float:
        return self.high_price

    @property
    def low(self) -> float:
        return self.low_price


class StockData(BaseModel):
    """单只股票的所有历史数据"""

    symbol: str = Field(..., description="股票代码")
    name: Optional[str] = Field(None, description="股票名称")
    bars: List[StockDailyBar] = Field(..., description="日线数据列表")
