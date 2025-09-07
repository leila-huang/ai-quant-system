"""
回测引擎模块

提供高性能的向量化回测功能，支持多策略并行、A股市场约束、
交易成本建模和专业回测报告生成等功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import date, datetime
from enum import Enum
import pandas as pd
import numpy as np

from backend.src.models.basic_models import StockData


class BacktestEngine(ABC):
    """回测引擎抽象基类"""
    
    @abstractmethod
    def run_backtest(self, strategy_config: Dict[str, Any], 
                    universe: List[str], 
                    start_date: date, 
                    end_date: date, **kwargs) -> 'BacktestResult':
        """
        执行回测
        
        Args:
            strategy_config: 策略配置
            universe: 股票池
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 额外参数
            
        Returns:
            BacktestResult: 回测结果
        """
        pass
    
    @abstractmethod
    def get_supported_strategies(self) -> List[str]:
        """
        获取支持的策略类型
        
        Returns:
            List[str]: 支持的策略列表
        """
        pass


class MarketConstraint(ABC):
    """市场约束抽象基类"""
    
    @abstractmethod
    def apply_constraint(self, signals: pd.DataFrame, 
                        market_data: pd.DataFrame) -> pd.DataFrame:
        """
        应用市场约束
        
        Args:
            signals: 交易信号
            market_data: 市场数据
            
        Returns:
            pd.DataFrame: 应用约束后的信号
        """
        pass


class CostModel(ABC):
    """交易成本模型抽象基类"""
    
    @abstractmethod
    def calculate_cost(self, trades: pd.DataFrame, 
                      market_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算交易成本
        
        Args:
            trades: 交易记录
            market_data: 市场数据
            
        Returns:
            pd.DataFrame: 包含成本的交易记录
        """
        pass
    
    @abstractmethod
    def get_cost_breakdown(self) -> Dict[str, str]:
        """
        获取成本组成说明
        
        Returns:
            Dict[str, str]: 成本组成字典
        """
        pass


class PerformanceCalculator(ABC):
    """业绩计算器抽象基类"""
    
    @abstractmethod
    def calculate_metrics(self, returns: pd.Series, 
                         benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        计算业绩指标
        
        Args:
            returns: 收益率序列
            benchmark: 基准收益率（可选）
            
        Returns:
            Dict[str, float]: 业绩指标字典
        """
        pass
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """
        获取支持的业绩指标
        
        Returns:
            List[str]: 支持的指标列表
        """
        pass


class BacktestResult:
    """回测结果类"""
    
    def __init__(self, 
                 returns: pd.Series,
                 positions: pd.DataFrame,
                 trades: pd.DataFrame,
                 metrics: Dict[str, float],
                 metadata: Dict[str, Any]):
        self.returns = returns
        self.positions = positions
        self.trades = trades
        self.metrics = metrics
        self.metadata = metadata
        
    def get_annual_return(self) -> float:
        """获取年化收益率"""
        return self.metrics.get('annual_return', 0.0)
        
    def get_sharpe_ratio(self) -> float:
        """获取夏普比率"""
        return self.metrics.get('sharpe_ratio', 0.0)
        
    def get_max_drawdown(self) -> float:
        """获取最大回撤"""
        return self.metrics.get('max_drawdown', 0.0)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'returns': self.returns.to_dict(),
            'positions': self.positions.to_dict('records'),
            'trades': self.trades.to_dict('records'),
            'metrics': self.metrics,
            'metadata': self.metadata,
        }


class ReportGenerator(ABC):
    """报告生成器抽象基类"""
    
    @abstractmethod
    def generate_report(self, backtest_result: BacktestResult, 
                       output_format: str = 'html') -> str:
        """
        生成回测报告
        
        Args:
            backtest_result: 回测结果
            output_format: 输出格式 ('html', 'pdf', 'json')
            
        Returns:
            str: 报告文件路径或内容
        """
        pass


# A股市场约束类型
class AStockConstraintType(Enum):
    """A股市场约束类型"""
    T_PLUS_1 = "t_plus_1"                  # T+1交易制度
    PRICE_LIMIT = "price_limit"            # 涨跌停限制
    SUSPENSION = "suspension"              # 停牌处理
    NEW_LISTING = "new_listing"            # 新股上市限制
    ST_STOCK = "st_stock"                  # ST股票限制


# 模块级常量
SUPPORTED_CONSTRAINTS = [
    "T+1_TRADING",      # T+1交易制度
    "PRICE_LIMITS",     # 涨跌停限制
    "SUSPENSION",       # 停牌处理
    "LIQUIDITY",        # 流动性约束
]

SUPPORTED_COST_TYPES = [
    "COMMISSION",       # 佣金
    "STAMP_TAX",        # 印花税
    "TRANSFER_FEE",     # 过户费
    "SLIPPAGE",         # 滑点成本
]

PERFORMANCE_METRICS = [
    "annual_return",    # 年化收益率
    "volatility",       # 年化波动率
    "sharpe_ratio",     # 夏普比率
    "calmar_ratio",     # 卡尔玛比率
    "max_drawdown",     # 最大回撤
    "win_rate",         # 胜率
    "profit_loss_ratio", # 盈亏比
]

__all__ = [
    "BacktestEngine",
    "MarketConstraint",
    "CostModel",
    "PerformanceCalculator",
    "BacktestResult",
    "ReportGenerator",
    "AStockConstraintType",
    "SUPPORTED_CONSTRAINTS",
    "SUPPORTED_COST_TYPES", 
    "PERFORMANCE_METRICS",
]



