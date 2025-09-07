"""
业绩指标计算模块

实现完整的量化投资业绩评估指标，包括收益率、风险指标、
风险调整收益指标等，为回测报告提供准确的指标计算。
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from backend.src.engine.backtest import PerformanceCalculator

logger = logging.getLogger(__name__)


class PeriodType(Enum):
    """时间周期类型"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class MetricCategory(Enum):
    """指标分类"""
    RETURN = "return"          # 收益指标
    RISK = "risk"             # 风险指标
    RISK_ADJUSTED = "risk_adjusted"  # 风险调整收益指标
    DRAWDOWN = "drawdown"     # 回撤指标
    TRADING = "trading"       # 交易指标
    DISTRIBUTION = "distribution"  # 分布指标


@dataclass
class PerformanceMetrics:
    """业绩指标数据类"""
    # 收益指标
    total_return: float = 0.0          # 总收益率
    annualized_return: float = 0.0     # 年化收益率
    cumulative_return: float = 0.0     # 累计收益率
    
    # 风险指标
    volatility: float = 0.0            # 年化波动率
    downside_volatility: float = 0.0   # 下行波动率
    tracking_error: float = 0.0        # 跟踪误差
    
    # 风险调整收益指标
    sharpe_ratio: float = 0.0          # 夏普比率
    sortino_ratio: float = 0.0         # 索提诺比率
    information_ratio: float = 0.0     # 信息比率
    calmar_ratio: float = 0.0          # 卡尔玛比率
    
    # 回撤指标
    max_drawdown: float = 0.0          # 最大回撤
    max_drawdown_duration: int = 0     # 最大回撤持续期
    avg_drawdown: float = 0.0          # 平均回撤
    recovery_factor: float = 0.0       # 恢复因子
    
    # 分布指标
    skewness: float = 0.0              # 偏度
    kurtosis: float = 0.0              # 峰度
    var_95: float = 0.0                # 95% VaR
    cvar_95: float = 0.0               # 95% CVaR
    
    # 交易指标
    win_rate: float = 0.0              # 胜率
    profit_loss_ratio: float = 0.0     # 盈亏比
    expectancy: float = 0.0            # 期望收益
    
    # 其他指标
    beta: Optional[float] = None       # 贝塔系数
    alpha: Optional[float] = None      # 阿尔法
    r_squared: Optional[float] = None  # R平方
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'volatility': self.volatility,
            'downside_volatility': self.downside_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_drawdown': self.avg_drawdown,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'win_rate': self.win_rate,
            'profit_loss_ratio': self.profit_loss_ratio,
            'expectancy': self.expectancy,
            'beta': self.beta,
            'alpha': self.alpha,
            'r_squared': self.r_squared
        }


class PerformanceAnalyzer(PerformanceCalculator):
    """
    业绩分析器
    
    实现完整的量化投资业绩评估指标计算，包括收益率、风险指标、
    风险调整收益指标等，提供专业的业绩分析能力。
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.03,
                 trading_days_per_year: int = 252):
        """
        初始化业绩分析器
        
        Args:
            risk_free_rate: 无风险利率（年化）
            trading_days_per_year: 年交易日数量
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        
        logger.info(f"业绩分析器初始化完成 - 无风险利率: {risk_free_rate:.2%}")
    
    def calculate_metrics(self, 
                         returns: pd.Series, 
                         benchmark: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        计算业绩指标
        
        Args:
            returns: 收益率序列（日收益率）
            benchmark: 基准收益率序列（可选）
            
        Returns:
            PerformanceMetrics: 业绩指标
        """
        try:
            if returns.empty:
                logger.warning("收益率序列为空")
                return PerformanceMetrics()
            
            metrics = PerformanceMetrics()
            
            # 基础数据清理
            returns = returns.dropna()
            if len(returns) == 0:
                logger.warning("清理后收益率序列为空")
                return metrics
            
            # 计算收益指标
            self._calculate_return_metrics(returns, metrics)
            
            # 计算风险指标
            self._calculate_risk_metrics(returns, metrics)
            
            # 计算风险调整收益指标
            self._calculate_risk_adjusted_metrics(returns, metrics)
            
            # 计算回撤指标
            self._calculate_drawdown_metrics(returns, metrics)
            
            # 计算分布指标
            self._calculate_distribution_metrics(returns, metrics)
            
            # 计算交易指标
            self._calculate_trading_metrics(returns, metrics)
            
            # 如果有基准，计算相对指标
            if benchmark is not None and not benchmark.empty:
                self._calculate_relative_metrics(returns, benchmark, metrics)
            
            logger.info("业绩指标计算完成")
            return metrics
            
        except Exception as e:
            logger.error(f"计算业绩指标失败: {e}")
            return PerformanceMetrics()
    
    def _calculate_return_metrics(self, returns: pd.Series, metrics: PerformanceMetrics):
        """计算收益指标"""
        try:
            # 累计收益率
            cumulative_returns = (1 + returns).cumprod()
            metrics.total_return = float(cumulative_returns.iloc[-1] - 1)
            metrics.cumulative_return = metrics.total_return
            
            # 年化收益率
            periods = len(returns)
            years = periods / self.trading_days
            if years > 0 and cumulative_returns.iloc[-1] > 0:
                metrics.annualized_return = float((cumulative_returns.iloc[-1]) ** (1 / years) - 1)
            else:
                metrics.annualized_return = 0.0
                
        except Exception as e:
            logger.warning(f"计算收益指标失败: {e}")
    
    def _calculate_risk_metrics(self, returns: pd.Series, metrics: PerformanceMetrics):
        """计算风险指标"""
        try:
            # 年化波动率
            metrics.volatility = float(returns.std() * np.sqrt(self.trading_days))
            
            # 下行波动率（只考虑负收益）
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                metrics.downside_volatility = float(negative_returns.std() * np.sqrt(self.trading_days))
            else:
                metrics.downside_volatility = 0.0
                
        except Exception as e:
            logger.warning(f"计算风险指标失败: {e}")
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series, metrics: PerformanceMetrics):
        """计算风险调整收益指标"""
        try:
            daily_rf_rate = self.risk_free_rate / self.trading_days
            
            # 夏普比率
            if metrics.volatility > 0:
                excess_return = metrics.annualized_return - self.risk_free_rate
                metrics.sharpe_ratio = float(excess_return / metrics.volatility)
            
            # 索提诺比率
            if metrics.downside_volatility > 0:
                excess_return = metrics.annualized_return - self.risk_free_rate
                metrics.sortino_ratio = float(excess_return / metrics.downside_volatility)
            
            # 卡尔玛比率
            if abs(metrics.max_drawdown) > 0:
                metrics.calmar_ratio = float(metrics.annualized_return / abs(metrics.max_drawdown))
                
        except Exception as e:
            logger.warning(f"计算风险调整收益指标失败: {e}")
    
    def _calculate_drawdown_metrics(self, returns: pd.Series, metrics: PerformanceMetrics):
        """计算回撤指标"""
        try:
            # 计算净值曲线
            cumulative_returns = (1 + returns).cumprod()
            
            # 计算回撤序列
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            # 最大回撤
            metrics.max_drawdown = float(drawdown.min())
            
            # 平均回撤
            negative_drawdown = drawdown[drawdown < 0]
            if len(negative_drawdown) > 0:
                metrics.avg_drawdown = float(negative_drawdown.mean())
            
            # 最大回撤持续期
            drawdown_periods = self._calculate_drawdown_duration(drawdown)
            if drawdown_periods:
                metrics.max_drawdown_duration = max(drawdown_periods)
            
            # 恢复因子
            if abs(metrics.max_drawdown) > 0:
                metrics.recovery_factor = float(metrics.total_return / abs(metrics.max_drawdown))
                
        except Exception as e:
            logger.warning(f"计算回撤指标失败: {e}")
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> List[int]:
        """计算回撤持续期"""
        durations = []
        in_drawdown = False
        duration = 0
        
        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    duration = 1
                else:
                    duration += 1
            else:
                if in_drawdown:
                    durations.append(duration)
                    in_drawdown = False
                    duration = 0
        
        # 如果序列结束时仍在回撤中
        if in_drawdown:
            durations.append(duration)
        
        return durations
    
    def _calculate_distribution_metrics(self, returns: pd.Series, metrics: PerformanceMetrics):
        """计算分布指标"""
        try:
            # 偏度
            metrics.skewness = float(returns.skew())
            
            # 峰度
            metrics.kurtosis = float(returns.kurtosis())
            
            # VaR (Value at Risk) 95%
            metrics.var_95 = float(returns.quantile(0.05))
            
            # CVaR (Conditional Value at Risk) 95%
            var_threshold = returns.quantile(0.05)
            tail_returns = returns[returns <= var_threshold]
            if len(tail_returns) > 0:
                metrics.cvar_95 = float(tail_returns.mean())
            else:
                metrics.cvar_95 = metrics.var_95
                
        except Exception as e:
            logger.warning(f"计算分布指标失败: {e}")
    
    def _calculate_trading_metrics(self, returns: pd.Series, metrics: PerformanceMetrics):
        """计算交易指标"""
        try:
            # 胜率
            positive_returns = returns[returns > 0]
            metrics.win_rate = float(len(positive_returns) / len(returns))
            
            # 盈亏比
            if len(positive_returns) > 0 and len(returns[returns < 0]) > 0:
                avg_win = positive_returns.mean()
                avg_loss = returns[returns < 0].mean()
                metrics.profit_loss_ratio = float(abs(avg_win / avg_loss))
            
            # 期望收益
            metrics.expectancy = float(returns.mean())
            
        except Exception as e:
            logger.warning(f"计算交易指标失败: {e}")
    
    def _calculate_relative_metrics(self, returns: pd.Series, benchmark: pd.Series, metrics: PerformanceMetrics):
        """计算相对基准的指标"""
        try:
            # 对齐数据
            aligned_data = pd.DataFrame({'returns': returns, 'benchmark': benchmark}).dropna()
            if aligned_data.empty:
                return
            
            portfolio_returns = aligned_data['returns']
            benchmark_returns = aligned_data['benchmark']
            
            # 跟踪误差
            excess_returns = portfolio_returns - benchmark_returns
            metrics.tracking_error = float(excess_returns.std() * np.sqrt(self.trading_days))
            
            # 信息比率
            if metrics.tracking_error > 0:
                excess_return_annual = excess_returns.mean() * self.trading_days
                metrics.information_ratio = float(excess_return_annual / metrics.tracking_error)
            
            # Beta系数
            if benchmark_returns.var() > 0:
                metrics.beta = float(np.cov(portfolio_returns, benchmark_returns)[0, 1] / benchmark_returns.var())
            
            # Alpha系数
            if metrics.beta is not None:
                benchmark_annual_return = (1 + benchmark_returns.mean()) ** self.trading_days - 1
                expected_return = self.risk_free_rate + metrics.beta * (benchmark_annual_return - self.risk_free_rate)
                metrics.alpha = float(metrics.annualized_return - expected_return)
            
            # R平方
            if len(portfolio_returns) > 1 and benchmark_returns.var() > 0:
                correlation = portfolio_returns.corr(benchmark_returns)
                metrics.r_squared = float(correlation ** 2) if not np.isnan(correlation) else None
                
        except Exception as e:
            logger.warning(f"计算相对指标失败: {e}")
    
    def get_supported_metrics(self) -> List[str]:
        """获取支持的指标列表"""
        return [
            'total_return', 'annualized_return', 'cumulative_return',
            'volatility', 'downside_volatility', 'tracking_error',
            'sharpe_ratio', 'sortino_ratio', 'information_ratio', 'calmar_ratio',
            'max_drawdown', 'max_drawdown_duration', 'avg_drawdown', 'recovery_factor',
            'skewness', 'kurtosis', 'var_95', 'cvar_95',
            'win_rate', 'profit_loss_ratio', 'expectancy',
            'beta', 'alpha', 'r_squared'
        ]
    
    def calculate_period_metrics(self, 
                               returns: pd.Series, 
                               period: PeriodType = PeriodType.MONTHLY) -> pd.DataFrame:
        """
        计算分期间业绩指标
        
        Args:
            returns: 收益率序列
            period: 分析周期
            
        Returns:
            pd.DataFrame: 分期间指标
        """
        try:
            # 根据期间类型重新采样
            if period == PeriodType.WEEKLY:
                period_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
            elif period == PeriodType.MONTHLY:
                period_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            elif period == PeriodType.QUARTERLY:
                period_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
            elif period == PeriodType.YEARLY:
                period_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
            else:
                period_returns = returns
            
            # 计算每期指标
            results = []
            for period_end, period_return in period_returns.items():
                period_data = returns.loc[returns.index <= period_end]
                if len(period_data) > 0:
                    period_metrics = self.calculate_metrics(period_data)
                    result = {
                        'period_end': period_end,
                        'period_return': period_return,
                        'cumulative_return': period_metrics.cumulative_return,
                        'volatility': period_metrics.volatility,
                        'sharpe_ratio': period_metrics.sharpe_ratio,
                        'max_drawdown': period_metrics.max_drawdown
                    }
                    results.append(result)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"计算分期间指标失败: {e}")
            return pd.DataFrame()
    
    def compare_strategies(self, 
                         strategies_returns: Dict[str, pd.Series],
                         benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        策略对比分析
        
        Args:
            strategies_returns: 策略收益率字典
            benchmark: 基准收益率
            
        Returns:
            pd.DataFrame: 对比分析结果
        """
        try:
            results = []
            
            for strategy_name, returns in strategies_returns.items():
                metrics = self.calculate_metrics(returns, benchmark)
                result = {
                    'strategy': strategy_name,
                    **metrics.to_dict()
                }
                results.append(result)
            
            # 如果有基准，也计算基准指标
            if benchmark is not None:
                benchmark_metrics = self.calculate_metrics(benchmark)
                benchmark_result = {
                    'strategy': 'Benchmark',
                    **benchmark_metrics.to_dict()
                }
                results.append(benchmark_result)
            
            comparison_df = pd.DataFrame(results)
            comparison_df = comparison_df.set_index('strategy')
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"策略对比分析失败: {e}")
            return pd.DataFrame()


def calculate_basic_metrics(returns: pd.Series, 
                          risk_free_rate: float = 0.03) -> Dict[str, float]:
    """
    计算基本业绩指标的便利函数
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        Dict[str, float]: 基本指标字典
    """
    analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)
    metrics = analyzer.calculate_metrics(returns)
    
    return {
        'Total Return': metrics.total_return,
        'Annualized Return': metrics.annualized_return,
        'Volatility': metrics.volatility,
        'Sharpe Ratio': metrics.sharpe_ratio,
        'Max Drawdown': metrics.max_drawdown,
        'Win Rate': metrics.win_rate
    }


def calculate_rolling_metrics(returns: pd.Series, 
                            window: int = 252,
                            metrics: List[str] = None) -> pd.DataFrame:
    """
    计算滚动业绩指标
    
    Args:
        returns: 收益率序列
        window: 滚动窗口大小（交易日）
        metrics: 要计算的指标列表
        
    Returns:
        pd.DataFrame: 滚动指标
    """
    if metrics is None:
        metrics = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
    
    analyzer = PerformanceAnalyzer()
    rolling_results = []
    
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        window_metrics = analyzer.calculate_metrics(window_returns)
        
        result = {
            'date': returns.index[i-1],
            'annualized_return': window_metrics.annualized_return,
            'volatility': window_metrics.volatility,
            'sharpe_ratio': window_metrics.sharpe_ratio,
            'max_drawdown': window_metrics.max_drawdown
        }
        rolling_results.append(result)
    
    rolling_df = pd.DataFrame(rolling_results)
    rolling_df = rolling_df.set_index('date')
    
    return rolling_df[metrics] if rolling_results else pd.DataFrame()



