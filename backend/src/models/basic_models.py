"""
基础数据模型 - 不使用Pydantic，避免版本兼容性问题
"""

from datetime import date
from typing import List, Optional
from dataclasses import dataclass
import pandas as pd


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
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        将StockData转换为pandas DataFrame
        
        Returns:
            pd.DataFrame: 包含OHLCV数据的DataFrame，按日期排序
        """
        if not self.bars:
            return pd.DataFrame()
        
        data = []
        for bar in self.bars:
            row = {
                'date': bar.date,
                'open': bar.open_price,
                'high': bar.high_price,
                'low': bar.low_price,
                'close': bar.close_price,
                'volume': bar.volume,
                'amount': bar.amount,
                'adjust_factor': bar.adjust_factor,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df['symbol'] = self.symbol
        df['name'] = self.name or ''
        
        # 确保日期列为datetime类型并排序
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
