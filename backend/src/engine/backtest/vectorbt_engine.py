"""
Vectorbt回测引擎

基于vectorbt库实现的高性能向量化回测引擎，支持多策略、
多股票的并行回测，具备完整的性能分析和风险评估功能。
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import time
import gc

import numpy as np
import pandas as pd
import vectorbt as vbt

from backend.src.models.basic_models import StockData
from backend.src.storage.parquet_engine import ParquetStorage
from backend.src.engine.backtest import BacktestEngine, BacktestResult
from backend.src.engine.backtest.constraints import (
    AStockConstraints, ConstraintConfig, create_default_a_stock_constraints
)
from backend.src.engine.backtest.cost_model import (
    TradingCostModel, CostConfig, create_standard_cost_model
)
from backend.src.engine.utils import PerformanceMonitor, ValidationHelper

# 禁用vectorbt的警告信息
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"        # 市价单
    LIMIT = "limit"          # 限价单
    STOP = "stop"           # 止损单
    STOP_LIMIT = "stop_limit" # 止损限价单


class PositionSizing(Enum):
    """仓位管理方法"""
    FIXED_AMOUNT = "fixed_amount"     # 固定金额
    FIXED_SHARES = "fixed_shares"     # 固定股数
    PERCENTAGE = "percentage"         # 固定百分比
    VOLATILITY = "volatility"         # 波动率调整
    KELLY = "kelly"                   # 凯利公式
    

@dataclass
class TradingConfig:
    """交易配置"""
    # 基础配置
    initial_cash: float = 1000000.0      # 初始资金
    commission: float = 0.0003           # 手续费率
    slippage: float = 0.001              # 滑点
    
    # 订单配置
    order_type: OrderType = OrderType.MARKET
    position_sizing: PositionSizing = PositionSizing.PERCENTAGE
    default_position_size: float = 0.05   # 默认仓位大小（5%）
    
    # 风险管理
    max_position_size: float = 0.2       # 最大单股仓位
    stop_loss: Optional[float] = 0.05    # 止损比例（5%）
    take_profit: Optional[float] = 0.1   # 止盈比例（10%）
    
    # 交易限制
    min_trade_amount: float = 1000       # 最小交易金额
    max_daily_trades: int = 50           # 每日最大交易次数
    
    # A股特殊规则
    t_plus_1: bool = True                # T+1交易制度
    price_limit: bool = True             # 涨跌停限制（10%）
    
    def __post_init__(self):
        """配置验证"""
        if self.initial_cash <= 0:
            raise ValueError("初始资金必须大于0")
        if not 0 <= self.commission <= 0.1:
            raise ValueError("手续费率应在0-10%之间")
        if not 0 <= self.slippage <= 0.1:
            raise ValueError("滑点应在0-10%之间")


@dataclass
class BacktestStrategy:
    """回测策略定义"""
    name: str                                    # 策略名称
    signal_func: Callable                       # 信号生成函数
    position_sizing_func: Optional[Callable] = None  # 仓位管理函数
    entry_conditions: Optional[List[str]] = None     # 入场条件
    exit_conditions: Optional[List[str]] = None      # 出场条件
    parameters: Optional[Dict[str, Any]] = None      # 策略参数
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class VectorbtBacktestEngine(BacktestEngine):
    """
    Vectorbt高性能回测引擎
    
    提供基于vectorbt的向量化回测，支持多策略、多股票的
    并行回测，具备完整的性能分析和风险管理功能。
    """
    
    def __init__(self, 
                 config: TradingConfig = None,
                 data_storage: ParquetStorage = None,
                 constraints: AStockConstraints = None,
                 enable_constraints: bool = True,
                 cost_model: TradingCostModel = None,
                 enable_cost_model: bool = True):
        """
        初始化vectorbt回测引擎
        
        Args:
            config: 交易配置
            data_storage: 数据存储引擎
            constraints: A股市场约束模型
            enable_constraints: 是否启用约束
            cost_model: 交易成本模型
            enable_cost_model: 是否启用成本计算
        """
        self.config = config or TradingConfig()
        self.data_storage = data_storage or ParquetStorage()
        
        # A股约束模型
        self.enable_constraints = enable_constraints
        self.constraints = constraints or (create_default_a_stock_constraints() if enable_constraints else None)
        
        # 交易成本模型
        self.enable_cost_model = enable_cost_model
        self.cost_model = cost_model or (create_standard_cost_model() if enable_cost_model else None)
        
        # 回测状态
        self.is_initialized = False
        self.strategies = {}
        self.backtest_results = {}
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 数据缓存
        self._price_cache = {}
        self._signal_cache = {}
        
        # vectorbt配置
        try:
            if hasattr(vbt.settings, 'caching'):
                vbt.settings.caching['enabled'] = True  # 启用缓存
            # 注意：chunking在此版本的vectorbt中不可用
        except Exception as e:
            logger.debug(f"设置vectorbt配置时出现警告: {e}")
        
        constraint_status = "启用A股约束" if self.enable_constraints else "禁用约束"
        cost_status = "启用成本计算" if self.enable_cost_model else "禁用成本计算"
        logger.info(f"Vectorbt回测引擎初始化完成 - {constraint_status}, {cost_status}")
        
        # 添加默认策略
        self._add_default_strategies()
    
    def _add_default_strategies(self):
        """添加默认策略"""
        try:
            # 添加双均线策略
            ma_strategy = create_simple_ma_strategy(fast_window=5, slow_window=20)
            self.add_strategy(ma_strategy)
            
            # 添加RSI策略
            rsi_strategy = create_rsi_strategy(rsi_window=14, oversold=30, overbought=70)
            self.add_strategy(rsi_strategy)
            
            # 添加动量策略
            momentum_strategy = self._create_momentum_strategy()
            self.add_strategy(momentum_strategy)
            
            # 建立策略类型映射
            self.strategy_type_mapping = {
                'ma_crossover': 'MA_5_20',
                'rsi_mean_reversion': 'RSI_14_30_70', 
                'momentum': 'Momentum_20_0.02'
            }
            
            logger.info("默认策略添加完成")
            
        except Exception as e:
            logger.error(f"添加默认策略失败: {e}")
    
    def _create_momentum_strategy(self, lookback_period: int = 20, threshold: float = 0.02) -> BacktestStrategy:
        """创建动量策略"""
        def momentum_signal_func(data: pd.DataFrame, 
                                lookback_period: int, 
                                threshold: float) -> pd.DataFrame:
            """动量信号生成函数"""
            # 获取收盘价
            if 'close' in data.columns:
                close_prices = data['close']
            else:
                close_prices = data.xs('close', axis=1, level=1) if data.columns.nlevels > 1 else data
            
            # 计算动量
            momentum = close_prices / close_prices.shift(lookback_period) - 1
            
            # 生成信号
            buy_signals = momentum > threshold
            sell_signals = momentum < -threshold
            
            # 创建信号矩阵
            signals = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            signals[buy_signals] = 1
            signals[sell_signals] = -1
            signals = signals.fillna(0)
            
            return signals
        
        return BacktestStrategy(
            name=f'Momentum_{lookback_period}_{threshold}',
            signal_func=momentum_signal_func,
            parameters={'lookback_period': lookback_period, 'threshold': threshold}
        )
    
    def add_strategy(self, strategy: BacktestStrategy) -> bool:
        """
        添加回测策略
        
        Args:
            strategy: 策略定义
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 验证策略
            if not strategy.name:
                raise ValueError("策略名称不能为空")
            
            if not callable(strategy.signal_func):
                raise ValueError("信号生成函数必须是可调用对象")
            
            # 存储策略
            self.strategies[strategy.name] = strategy
            
            logger.info(f"策略 '{strategy.name}' 添加成功")
            return True
            
        except Exception as e:
            logger.error(f"添加策略失败: {e}")
            return False
    
    def get_strategy_name_by_type(self, strategy_type: str) -> str:
        """根据策略类型获取内部策略名称"""
        return self.strategy_type_mapping.get(strategy_type, strategy_type)
    
    def load_data(self, 
                  symbols: Union[str, List[str]], 
                  start_date: date, 
                  end_date: date,
                  fields: List[str] = None) -> pd.DataFrame:
        """
        加载回测数据
        
        Args:
            symbols: 股票代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要的字段列表
            
        Returns:
            pd.DataFrame: 回测数据
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume']
        
        logger.info(f"加载回测数据: {len(symbols)}只股票，{start_date} 到 {end_date}")
        
        try:
            all_data = []
            
            for symbol in symbols:
                # 检查缓存
                cache_key = f"{symbol}_{start_date}_{end_date}"
                if cache_key in self._price_cache:
                    symbol_data = self._price_cache[cache_key]
                else:
                    # 从存储加载数据
                    stock_data = self.data_storage.load_stock_data(
                        symbol, start_date, end_date
                    )
                    
                    if not stock_data or len(stock_data.bars) == 0:
                        logger.warning(f"股票 {symbol} 没有数据")
                        continue
                    
                    # 转换为DataFrame格式
                    symbol_data = stock_data.to_dataframe()
                    symbol_data = symbol_data[symbol_data['date'].between(start_date, end_date)]
                    
                    # 缓存数据
                    self._price_cache[cache_key] = symbol_data
                
                if len(symbol_data) > 0:
                    symbol_data['symbol'] = symbol
                    all_data.append(symbol_data)
            
            if not all_data:
                raise ValueError("没有加载到任何数据")
            
            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # 创建多层索引的价格矩阵
            price_matrix = combined_data.pivot_table(
                index='date',
                columns='symbol',
                values=fields
            )
            
            # 确保日期索引排序
            price_matrix = price_matrix.sort_index()
            
            logger.info(f"数据加载完成: {price_matrix.shape}")
            return price_matrix
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def generate_signals(self, 
                        data: pd.DataFrame, 
                        strategy_name: str) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 价格数据
            strategy_name: 策略名称
            
        Returns:
            pd.DataFrame: 交易信号
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"策略 '{strategy_name}' 不存在")
        
        strategy = self.strategies[strategy_name]
        
        try:
            logger.info(f"生成 '{strategy_name}' 策略信号...")
            
            # 调用策略的信号生成函数
            signals = strategy.signal_func(data, **strategy.parameters)
            
            # 确保信号格式正确
            if not isinstance(signals, pd.DataFrame):
                raise ValueError("策略信号必须返回DataFrame格式")
            
            # 验证信号值（应该是布尔值或0/1）
            for col in signals.columns:
                if not signals[col].dtype in ['bool', 'int8', 'int64']:
                    signals[col] = signals[col].astype(bool)
            
            # 应用A股市场约束
            if self.enable_constraints and self.constraints:
                logger.info("应用A股市场约束...")
                try:
                    constrained_signals = self.constraints.apply_constraint(signals, data)
                    logger.info(f"约束应用完成: {signals.sum().sum()} -> {constrained_signals.sum().sum()}")
                    signals = constrained_signals
                except Exception as e:
                    logger.warning(f"应用A股约束失败，使用原始信号: {e}")
            
            logger.info(f"策略信号生成完成: {signals.shape}")
            return signals
            
        except Exception as e:
            logger.error(f"生成策略信号失败: {e}")
            raise
    
    def calculate_position_sizes(self, 
                               signals: pd.DataFrame, 
                               data: pd.DataFrame,
                               strategy_name: str) -> pd.DataFrame:
        """
        计算仓位大小
        
        Args:
            signals: 交易信号
            data: 价格数据
            strategy_name: 策略名称
            
        Returns:
            pd.DataFrame: 仓位大小
        """
        strategy = self.strategies[strategy_name]
        
        try:
            # 使用策略自定义仓位管理函数
            if strategy.position_sizing_func:
                position_sizes = strategy.position_sizing_func(
                    signals, data, self.config, **strategy.parameters
                )
            else:
                # 使用默认仓位管理
                position_sizes = self._default_position_sizing(signals, data)
            
            return position_sizes
            
        except Exception as e:
            logger.error(f"计算仓位大小失败: {e}")
            raise
    
    def _default_position_sizing(self, 
                               signals: pd.DataFrame, 
                               data: pd.DataFrame) -> pd.DataFrame:
        """默认仓位管理策略"""
        position_sizes = signals.copy()
        
        if self.config.position_sizing == PositionSizing.PERCENTAGE:
            # 固定百分比仓位
            position_sizes = position_sizes * self.config.default_position_size
        elif self.config.position_sizing == PositionSizing.FIXED_AMOUNT:
            # 固定金额仓位
            close_prices = data['close'] if 'close' in data.columns else data.iloc[:, -1]
            shares = self.config.default_position_size / close_prices
            position_sizes = signals * shares.reindex(signals.index)
        else:
            # 其他复杂仓位管理策略待实现
            position_sizes = position_sizes * self.config.default_position_size
        
        # 应用最大仓位限制
        position_sizes = position_sizes.clip(upper=self.config.max_position_size)
        
        return position_sizes
    
    def run_backtest(self, 
                    symbols: Union[str, List[str]],
                    start_date: date,
                    end_date: date,
                    strategy_names: Union[str, List[str]] = None) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            symbols: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            strategy_names: 策略名称列表
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        start_time = time.time()
        
        try:
            # 参数处理
            if isinstance(symbols, str):
                symbols = [symbols]
            if isinstance(strategy_names, str):
                strategy_names = [strategy_names]
            if strategy_names is None:
                strategy_names = list(self.strategies.keys())
            
            if not strategy_names:
                raise ValueError("没有可用的策略")
            
            logger.info(f"开始回测: {len(symbols)}只股票, {len(strategy_names)}个策略")
            
            # 加载数据
            data = self.load_data(symbols, start_date, end_date)
            
            if data.empty:
                raise ValueError("没有可用的价格数据")
            
            backtest_results = {}
            
            # 为每个策略运行回测
            for strategy_name in strategy_names:
                try:
                    logger.info(f"回测策略: {strategy_name}")
                    
                    # 生成交易信号
                    signals = self.generate_signals(data, strategy_name)
                    
                    # 计算仓位大小
                    position_sizes = self.calculate_position_sizes(signals, data, strategy_name)
                    
                    # 执行vectorbt回测
                    portfolio = self._run_vectorbt_backtest(
                        data, signals, position_sizes, strategy_name
                    )
                    
                    # 计算性能指标
                    performance_stats = self._calculate_performance_stats(portfolio)
                    
                    # 存储结果
                    strategy_result = {
                        'strategy_name': strategy_name,
                        'portfolio': portfolio,
                        'performance_stats': performance_stats,
                        'symbols': symbols,
                        'start_date': start_date,
                        'end_date': end_date,
                        'signals': signals,
                        'position_sizes': position_sizes
                    }
                    
                    backtest_results[strategy_name] = strategy_result
                    
                except Exception as e:
                    logger.error(f"策略 {strategy_name} 回测失败: {e}")
                    backtest_results[strategy_name] = {'error': str(e)}
            
            # 汇总结果
            summary = {
                'total_strategies': len(strategy_names),
                'successful_strategies': len([r for r in backtest_results.values() if 'error' not in r]),
                'symbols': symbols,
                'date_range': (start_date, end_date),
                'execution_time': time.time() - start_time,
                'data_shape': data.shape,
                'results': backtest_results
            }
            
            self.backtest_results[f"{'-'.join(strategy_names)}_{start_date}_{end_date}"] = summary
            
            logger.info(f"回测完成，耗时: {summary['execution_time']:.2f}秒")
            return summary
            
        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            raise
        finally:
            # 清理内存
            gc.collect()
    
    def _run_vectorbt_backtest(self, 
                              data: pd.DataFrame,
                              signals: pd.DataFrame, 
                              position_sizes: pd.DataFrame,
                              strategy_name: str) -> vbt.Portfolio:
        """
        执行vectorbt回测
        
        Args:
            data: 价格数据
            signals: 交易信号
            position_sizes: 仓位大小
            strategy_name: 策略名称
            
        Returns:
            vbt.Portfolio: 回测组合结果
        """
        try:
            # 获取价格数据
            if 'close' in data.columns:
                close_prices = data['close']
            else:
                # 如果是多层索引，取close价格
                close_prices = data.xs('close', axis=1, level=1) if data.columns.nlevels > 1 else data
            
            # 确保数据对齐
            close_prices = close_prices.reindex(signals.index).ffill()
            
            # 创建买入和卖出信号
            buy_signals = signals > 0
            sell_signals = signals < 0
            
            # 确保仓位大小为正数
            abs_position_sizes = position_sizes.abs()
            
            # 创建vectorbt组合
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=buy_signals,
                exits=sell_signals,
                size=abs_position_sizes,
                init_cash=self.config.initial_cash,
                fees=self.config.commission,
                slippage=self.config.slippage,
                freq='D'  # 日频数据
            )
            
            return portfolio
            
        except Exception as e:
            logger.error(f"vectorbt回测执行失败: {e}")
            raise
    
    def _calculate_performance_stats(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """计算性能统计指标"""
        try:
            stats = {}
            
            # 基础收益指标
            try:
                stats['total_return'] = float(portfolio.total_return()) if hasattr(portfolio, 'total_return') else 0.0
            except:
                stats['total_return'] = 0.0
                
            try:
                stats['annualized_return'] = float(portfolio.annualized_return()) if hasattr(portfolio, 'annualized_return') else 0.0
            except:
                stats['annualized_return'] = 0.0
                
            try:
                stats['max_drawdown'] = float(portfolio.max_drawdown()) if hasattr(portfolio, 'max_drawdown') else 0.0
            except:
                stats['max_drawdown'] = 0.0
            
            # 风险指标（这些在新版本中可能不可用）
            try:
                if hasattr(portfolio, 'returns'):
                    returns = portfolio.returns()
                    stats['volatility'] = float(returns.std() * np.sqrt(252))  # 年化波动率
                    risk_free_rate = 0.02
                    excess_returns = returns.mean() - risk_free_rate/252
                    stats['sharpe_ratio'] = float(excess_returns / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0.0
                else:
                    stats['volatility'] = 0.0
                    stats['sharpe_ratio'] = 0.0
            except:
                stats['volatility'] = 0.0
                stats['sharpe_ratio'] = 0.0
            
            try:
                if stats['max_drawdown'] != 0:
                    stats['calmar_ratio'] = stats['annualized_return'] / abs(stats['max_drawdown'])
                else:
                    stats['calmar_ratio'] = 0.0
            except:
                stats['calmar_ratio'] = 0.0
            
            # 交易统计
            try:
                if hasattr(portfolio, 'trades'):
                    trades = portfolio.trades
                    stats['total_trades'] = int(trades.count()) if hasattr(trades, 'count') else 0
                    stats['win_rate'] = float(trades.win_rate()) if hasattr(trades, 'win_rate') else 0.0
                    stats['profit_factor'] = float(trades.profit_factor()) if hasattr(trades, 'profit_factor') else 0.0
                    
                    if hasattr(trades, 'returns'):
                        trade_returns = trades.returns
                        stats['avg_trade_return'] = float(trade_returns.mean()) if len(trade_returns) > 0 else 0.0
                    else:
                        stats['avg_trade_return'] = 0.0
                else:
                    stats['total_trades'] = 0
                    stats['win_rate'] = 0.0
                    stats['profit_factor'] = 0.0
                    stats['avg_trade_return'] = 0.0
            except Exception as trade_error:
                logger.debug(f"交易统计计算失败: {trade_error}")
                stats['total_trades'] = 0
                stats['win_rate'] = 0.0
                stats['profit_factor'] = 0.0
                stats['avg_trade_return'] = 0.0
            
            # 持仓统计
            try:
                if hasattr(portfolio, 'cash'):
                    cash_series = portfolio.cash()
                    stats['max_cash'] = float(cash_series.max())
                    stats['min_cash'] = float(cash_series.min())
                else:
                    stats['max_cash'] = self.config.initial_cash
                    stats['min_cash'] = 0.0
                    
                if hasattr(portfolio, 'value'):
                    value_series = portfolio.value()
                    stats['final_value'] = float(value_series.iloc[-1])
                else:
                    stats['final_value'] = self.config.initial_cash
            except:
                stats['max_cash'] = self.config.initial_cash
                stats['min_cash'] = 0.0
                stats['final_value'] = self.config.initial_cash
            
            return stats
            
        except Exception as e:
            logger.warning(f"计算性能指标失败: {e}")
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'max_cash': self.config.initial_cash,
                'min_cash': 0.0,
                'final_value': self.config.initial_cash
            }
    
    def get_backtest_summary(self, result_key: str = None) -> Dict[str, Any]:
        """
        获取回测结果摘要
        
        Args:
            result_key: 结果键名，如果为None则返回最新结果
            
        Returns:
            Dict[str, Any]: 回测摘要
        """
        if not self.backtest_results:
            return {'error': '没有可用的回测结果'}
        
        if result_key is None:
            # 返回最新结果
            result_key = list(self.backtest_results.keys())[-1]
        
        if result_key not in self.backtest_results:
            return {'error': f'结果键 {result_key} 不存在'}
        
        result = self.backtest_results[result_key]
        
        # 创建摘要
        summary = {
            'meta': {
                'strategies': result['total_strategies'],
                'successful': result['successful_strategies'],
                'symbols': result['symbols'],
                'date_range': result['date_range'],
                'execution_time': result['execution_time']
            },
            'strategies': {}
        }
        
        # 为每个成功的策略添加摘要
        for strategy_name, strategy_result in result['results'].items():
            if 'error' not in strategy_result:
                perf = strategy_result['performance_stats']
                summary['strategies'][strategy_name] = {
                    'total_return': perf.get('total_return', 0),
                    'annualized_return': perf.get('annualized_return', 0),
                    'max_drawdown': perf.get('max_drawdown', 0),
                    'sharpe_ratio': perf.get('sharpe_ratio', 0),
                    'total_trades': perf.get('total_trades', 0),
                    'win_rate': perf.get('win_rate', 0)
                }
            else:
                summary['strategies'][strategy_name] = {'error': strategy_result['error']}
        
        return summary
    
    def clear_cache(self):
        """清理数据缓存"""
        self._price_cache.clear()
        self._signal_cache.clear()
        gc.collect()
        logger.info("缓存已清理")
    
    def get_supported_features(self) -> List[str]:
        """获取支持的功能列表"""
        return [
            '多策略并行回测',
            '多股票组合回测',
            'vectorbt向量化计算',
            '灵活仓位管理',
            '完整性能分析',
            'A股交易规则支持',
            '内存优化和缓存',
            '回测进度监控'
        ]
    
    # 实现抽象基类方法
    def run_backtest(self, strategy_config: Dict[str, Any], 
                    universe: List[str], 
                    start_date: date, 
                    end_date: date, **kwargs) -> BacktestResult:
        """
        执行回测 - 抽象基类接口实现
        
        Args:
            strategy_config: 策略配置字典
            universe: 股票池
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 额外参数
            
        Returns:
            BacktestResult: 标准化的回测结果
        """
        try:
            # 从策略配置中提取策略名称
            strategy_names = strategy_config.get('strategies', list(self.strategies.keys()))
            
            # 调用具体的回测实现
            detailed_results = self.run_vectorbt_backtest(
                symbols=universe,
                start_date=start_date,
                end_date=end_date,
                strategy_names=strategy_names
            )
            
            # 转换为标准化的BacktestResult格式
            if detailed_results['successful_strategies'] > 0:
                # 取第一个成功策略的结果作为主要结果
                first_success = None
                for result in detailed_results['results'].values():
                    if 'error' not in result:
                        first_success = result
                        break
                
                if first_success:
                    portfolio = first_success['portfolio']
                    
                    # 构造标准化结果
                    returns = portfolio.returns() if hasattr(portfolio, 'returns') else pd.Series()
                    positions = portfolio.positions if hasattr(portfolio, 'positions') else pd.DataFrame()
                    trades = portfolio.trades.records_readable if hasattr(portfolio, 'trades') else pd.DataFrame()
                    
                    return BacktestResult(
                        returns=returns,
                        positions=positions,
                        trades=trades,
                        metrics=first_success['performance_stats'],
                        metadata={
                            'engine': 'vectorbt',
                            'strategy_config': strategy_config,
                            'universe': universe,
                            'date_range': (start_date, end_date),
                            'execution_time': detailed_results['execution_time']
                        }
                    )
            
            # 如果没有成功结果，返回空的BacktestResult
            return BacktestResult(
                returns=pd.Series(),
                positions=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics={},
                metadata={'error': 'No successful backtest results'}
            )
            
        except Exception as e:
            logger.error(f"抽象接口回测执行失败: {e}")
            return BacktestResult(
                returns=pd.Series(),
                positions=pd.DataFrame(), 
                trades=pd.DataFrame(),
                metrics={},
                metadata={'error': str(e)}
            )
    
    def run_vectorbt_backtest(self, 
                    symbols: Union[str, List[str]],
                    start_date: date,
                    end_date: date,
                    strategy_names: Union[str, List[str]] = None) -> Dict[str, Any]:
        """
        运行vectorbt回测 - 具体实现方法（重命名原方法）
        
        Args:
            symbols: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            strategy_names: 策略名称列表
            
        Returns:
            Dict[str, Any]: 详细的回测结果
        """
        # 原run_backtest方法的实现保持不变
        start_time = time.time()
        
        try:
            # 参数处理
            if isinstance(symbols, str):
                symbols = [symbols]
            if isinstance(strategy_names, str):
                strategy_names = [strategy_names]
            if strategy_names is None:
                strategy_names = list(self.strategies.keys())
            
            if not strategy_names:
                raise ValueError("没有可用的策略")
            
            logger.info(f"开始回测: {len(symbols)}只股票, {len(strategy_names)}个策略")
            
            # 加载数据
            data = self.load_data(symbols, start_date, end_date)
            
            if data.empty:
                raise ValueError("没有可用的价格数据")
            
            backtest_results = {}
            
            # 为每个策略运行回测
            for strategy_name in strategy_names:
                try:
                    logger.info(f"回测策略: {strategy_name}")
                    
                    # 生成交易信号
                    signals = self.generate_signals(data, strategy_name)
                    
                    # 计算仓位大小
                    position_sizes = self.calculate_position_sizes(signals, data, strategy_name)
                    
                    # 执行vectorbt回测
                    portfolio = self._run_vectorbt_backtest(
                        data, signals, position_sizes, strategy_name
                    )
                    
                    # 计算性能指标
                    performance_stats = self._calculate_performance_stats(portfolio)
                    
                    # 存储结果
                    strategy_result = {
                        'strategy_name': strategy_name,
                        'portfolio': portfolio,
                        'performance_stats': performance_stats,
                        'symbols': symbols,
                        'start_date': start_date,
                        'end_date': end_date,
                        'signals': signals,
                        'position_sizes': position_sizes
                    }
                    
                    backtest_results[strategy_name] = strategy_result
                    
                except Exception as e:
                    logger.error(f"策略 {strategy_name} 回测失败: {e}")
                    backtest_results[strategy_name] = {'error': str(e)}
            
            # 汇总结果
            summary = {
                'total_strategies': len(strategy_names),
                'successful_strategies': len([r for r in backtest_results.values() if 'error' not in r]),
                'symbols': symbols,
                'date_range': (start_date, end_date),
                'execution_time': time.time() - start_time,
                'data_shape': data.shape,
                'results': backtest_results
            }
            
            self.backtest_results[f"{'-'.join(strategy_names)}_{start_date}_{end_date}"] = summary
            
            logger.info(f"回测完成，耗时: {summary['execution_time']:.2f}秒")
            return summary
            
        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            raise
        finally:
            # 清理内存
            gc.collect()
    
    def get_supported_strategies(self) -> List[str]:
        """获取支持的策略类型"""
        return list(self.strategies.keys())
    
    # A股约束管理方法
    def add_stock_info(self, symbol: str, market_type: str, listing_date: date = None, **kwargs):
        """
        添加股票信息到约束模型
        
        Args:
            symbol: 股票代码
            market_type: 市场类型 (main_board, chinext, star_market等)
            listing_date: 上市日期
            **kwargs: 其他股票属性
        """
        if not self.constraints:
            logger.warning("约束模型未启用，无法添加股票信息")
            return
        
        from backend.src.engine.backtest.constraints import StockInfo, MarketType, StockStatus
        
        # 转换市场类型
        market_type_enum = getattr(MarketType, market_type.upper(), MarketType.MAIN_BOARD)
        
        # 确定股票状态
        status = StockStatus.NORMAL
        if 'ST' in symbol or kwargs.get('is_st', False):
            status = StockStatus.ST
        if '*ST' in symbol:
            status = StockStatus.STAR_ST
        
        stock_info = StockInfo(
            symbol=symbol,
            name=kwargs.get('name', f'股票{symbol}'),
            market_type=market_type_enum,
            listing_date=listing_date,
            status=status,
            is_st='ST' in symbol
        )
        
        self.constraints.add_stock_info(stock_info)
        logger.info(f"已添加股票信息: {symbol} ({market_type})")
    
    def add_suspension_info(self, symbol: str, suspension_dates: List[date]):
        """
        添加停牌信息
        
        Args:
            symbol: 股票代码
            suspension_dates: 停牌日期列表
        """
        if not self.constraints:
            logger.warning("约束模型未启用，无法添加停牌信息")
            return
        
        self.constraints.add_suspension_dates(symbol, suspension_dates)
        logger.info(f"已添加停牌信息: {symbol} ({len(suspension_dates)} 天)")
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """获取约束模型信息"""
        if not self.constraints:
            return {'enabled': False, 'message': '约束模型未启用'}
        
        constraint_info = self.constraints.get_constraint_info()
        constraint_info['enabled'] = True
        return constraint_info
    
    def set_constraint_config(self, **config_updates):
        """
        更新约束配置
        
        Args:
            **config_updates: 配置更新字典
        """
        if not self.constraints:
            logger.warning("约束模型未启用，无法更新配置")
            return
        
        # 更新配置
        for key, value in config_updates.items():
            if hasattr(self.constraints.config, key):
                setattr(self.constraints.config, key, value)
                logger.info(f"约束配置更新: {key} = {value}")
            else:
                logger.warning(f"未知约束配置项: {key}")
    
    def disable_constraints(self):
        """禁用约束模型"""
        self.enable_constraints = False
        logger.info("已禁用A股约束模型")
    
    def enable_constraints(self):
        """启用约束模型"""
        if not self.constraints:
            self.constraints = create_default_a_stock_constraints()
        self.enable_constraints = True
        logger.info("已启用A股约束模型")
    
    # 成本模型管理方法
    def get_cost_model_info(self) -> Dict[str, Any]:
        """获取成本模型信息"""
        if not self.cost_model:
            return {'enabled': False, 'message': '成本模型未启用'}
        
        cost_info = self.cost_model.get_model_info()
        cost_info['enabled'] = True
        cost_info['cost_breakdown'] = self.cost_model.get_cost_breakdown()
        return cost_info
    
    def set_cost_model_config(self, **config_updates):
        """
        更新成本模型配置
        
        Args:
            **config_updates: 配置更新字典
        """
        if not self.cost_model:
            logger.warning("成本模型未启用，无法更新配置")
            return
        
        # 更新配置
        for key, value in config_updates.items():
            if hasattr(self.cost_model.config, key):
                setattr(self.cost_model.config, key, value)
                logger.info(f"成本配置更新: {key} = {value}")
            else:
                logger.warning(f"未知成本配置项: {key}")
    
    def generate_cost_analysis_report(self) -> Dict[str, Any]:
        """生成成本分析报告"""
        if not self.cost_model:
            return {'error': '成本模型未启用'}
        
        return self.cost_model.generate_cost_analysis_report()
    
    def estimate_cost_impact(self, strategy_returns: pd.Series, turnover_rate: float = 2.0) -> Dict[str, float]:
        """
        估算成本对收益的影响
        
        Args:
            strategy_returns: 策略收益率序列
            turnover_rate: 年换手率
            
        Returns:
            Dict[str, float]: 成本影响分析
        """
        if not self.cost_model:
            return {'error': '成本模型未启用'}
        
        return self.cost_model.estimate_cost_impact_on_returns(strategy_returns, turnover_rate)
    
    def disable_cost_model(self):
        """禁用成本模型"""
        self.enable_cost_model = False
        logger.info("已禁用交易成本模型")
    
    def enable_cost_model_func(self):
        """启用成本模型"""
        if not self.cost_model:
            self.cost_model = create_standard_cost_model()
        self.enable_cost_model = True
        logger.info("已启用交易成本模型")


def create_simple_ma_strategy(fast_window: int = 5, slow_window: int = 20) -> BacktestStrategy:
    """
    创建简单移动平均策略
    
    Args:
        fast_window: 快速移动平均窗口
        slow_window: 慢速移动平均窗口
        
    Returns:
        BacktestStrategy: 策略定义
    """
    def ma_signal_func(data: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
        """移动平均信号生成函数"""
        # 获取收盘价
        if 'close' in data.columns:
            close_prices = data['close']
        else:
            close_prices = data.xs('close', axis=1, level=1) if data.columns.nlevels > 1 else data
        
        # 计算移动平均
        fast_ma = close_prices.rolling(window=fast_window).mean()
        slow_ma = close_prices.rolling(window=slow_window).mean()
        
        # 生成信号：快线上穿慢线买入，下穿卖出
        buy_signals = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        sell_signals = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # 创建信号矩阵
        signals = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        signals = signals.fillna(0)
        
        return signals
    
    return BacktestStrategy(
        name=f'MA_{fast_window}_{slow_window}',
        signal_func=ma_signal_func,
        parameters={'fast_window': fast_window, 'slow_window': slow_window}
    )


def create_rsi_strategy(rsi_window: int = 14, oversold: int = 30, overbought: int = 70) -> BacktestStrategy:
    """
    创建RSI策略
    
    Args:
        rsi_window: RSI计算窗口
        oversold: 超卖阈值
        overbought: 超买阈值
        
    Returns:
        BacktestStrategy: RSI策略定义
    """
    def rsi_signal_func(data: pd.DataFrame, 
                       rsi_window: int, 
                       oversold: int, 
                       overbought: int) -> pd.DataFrame:
        """RSI信号生成函数"""
        # 获取收盘价
        if 'close' in data.columns:
            close_prices = data['close']
        else:
            close_prices = data.xs('close', axis=1, level=1) if data.columns.nlevels > 1 else data
        
        # 计算RSI
        def calculate_rsi(prices, window):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = close_prices.apply(lambda x: calculate_rsi(x, rsi_window))
        
        # 生成信号
        buy_signals = (rsi < oversold) & (rsi.shift(1) >= oversold)
        sell_signals = (rsi > overbought) & (rsi.shift(1) <= overbought)
        
        # 创建信号矩阵
        signals = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        signals = signals.fillna(0)
        
        return signals
    
    return BacktestStrategy(
        name=f'RSI_{rsi_window}_{oversold}_{overbought}',
        signal_func=rsi_signal_func,
        parameters={
            'rsi_window': rsi_window, 
            'oversold': oversold, 
            'overbought': overbought
        }
    )
