"""
工具模块

提供engine模块内部使用的各种工具函数和辅助类，包括性能监控、
数据转换、日期处理、缓存管理等通用功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar
from datetime import date, datetime, timedelta
import time
import logging
from functools import wraps
import pandas as pd
import numpy as np

from backend.src.models.basic_models import StockData

# 类型变量
T = TypeVar('T')


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: Dict[str, List[float]] = {}
        
    def time_function(self, func_name: str):
        """函数执行时间装饰器"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self._record_metric(func_name, execution_time)
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error(f"Function {func_name} failed after {execution_time:.4f}s: {e}")
                    raise
            return wrapper
        return decorator
        
    def _record_metric(self, name: str, value: float):
        """记录性能指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        # 记录日志
        self.logger.info(f"Performance - {name}: {value:.4f}s")
        
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取性能统计"""
        if name not in self.metrics:
            return {}
            
        values = self.metrics[name]
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
        }


class DataConverter:
    """数据转换工具"""
    
    @staticmethod
    def stock_data_to_dataframe(stock_data: StockData) -> pd.DataFrame:
        """
        将StockData转换为DataFrame
        
        Args:
            stock_data: StockData对象
            
        Returns:
            pd.DataFrame: 转换后的DataFrame
        """
        data = []
        for bar in stock_data.bars:
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
        if not df.empty:
            df['symbol'] = stock_data.symbol
            df['name'] = stock_data.name or ''
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def ensure_dataframe(data: Union[StockData, pd.DataFrame]) -> pd.DataFrame:
        """
        确保输入数据为DataFrame格式
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: DataFrame格式数据
        """
        if isinstance(data, StockData):
            return DataConverter.stock_data_to_dataframe(data)
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")


class DateHelper:
    """日期处理辅助工具"""
    
    @staticmethod
    def get_trading_days(start_date: date, end_date: date) -> List[date]:
        """
        获取交易日列表（简化版，实际应结合交易日历）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[date]: 交易日列表
        """
        days = []
        current_date = start_date
        
        while current_date <= end_date:
            # 简化版：排除周末，实际应查询交易日历
            if current_date.weekday() < 5:  # 周一到周五
                days.append(current_date)
            current_date += timedelta(days=1)
            
        return days
    
    @staticmethod
    def is_trading_day(target_date: date) -> bool:
        """
        检查是否为交易日（简化版）
        
        Args:
            target_date: 目标日期
            
        Returns:
            bool: 是否为交易日
        """
        # 简化版：仅排除周末
        return target_date.weekday() < 5
    
    @staticmethod
    def get_previous_trading_day(target_date: date) -> date:
        """
        获取前一个交易日
        
        Args:
            target_date: 目标日期
            
        Returns:
            date: 前一个交易日
        """
        current_date = target_date - timedelta(days=1)
        while not DateHelper.is_trading_day(current_date):
            current_date -= timedelta(days=1)
        return current_date


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.max_size = max_size
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
            
        self.cache[key] = value
        self.access_times[key] = datetime.now()
        
    def _evict_lru(self) -> None:
        """驱逐最近最少使用的条目"""
        if not self.cache:
            return
            
        # 找到最少使用的key
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: self.access_times[k])
        
        # 删除缓存项
        del self.cache[lru_key]
        del self.access_times[lru_key]
        
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


class ValidationHelper:
    """数据验证辅助工具"""
    
    @staticmethod
    def validate_dataframe_columns(df: pd.DataFrame, 
                                  required_columns: List[str]) -> bool:
        """
        验证DataFrame是否包含必需的列
        
        Args:
            df: DataFrame
            required_columns: 必需的列名列表
            
        Returns:
            bool: 是否包含所有必需列
        """
        return all(col in df.columns for col in required_columns)
    
    @staticmethod
    def validate_date_range(start_date: date, end_date: date) -> bool:
        """
        验证日期范围是否有效
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 日期范围是否有效
        """
        return start_date <= end_date
    
    @staticmethod
    def validate_stock_symbols(symbols: List[str]) -> List[str]:
        """
        验证股票代码格式
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            List[str]: 有效的股票代码列表
        """
        valid_symbols = []
        for symbol in symbols:
            # 简化验证：6位数字
            if isinstance(symbol, str) and symbol.isdigit() and len(symbol) == 6:
                valid_symbols.append(symbol)
        
        return valid_symbols


# 模块级实例
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager()

__all__ = [
    "PerformanceMonitor",
    "DataConverter",
    "DateHelper", 
    "CacheManager",
    "ValidationHelper",
    "performance_monitor",
    "cache_manager",
]



