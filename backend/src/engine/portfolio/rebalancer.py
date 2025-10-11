"""
投资组合再平衡器

实现投资组合的再平衡逻辑，支持：
- 多种再平衡频率（每日、每周、每月）
- 再平衡阈值触发
- 交易成本考虑
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """再平衡频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


class RebalanceTrigger(Enum):
    """再平衡触发方式"""
    SCHEDULED = "scheduled"          # 定期触发
    THRESHOLD = "threshold"          # 阈值触发
    HYBRID = "hybrid"                # 混合触发


@dataclass
class RebalanceConfig:
    """再平衡配置"""
    # 频率设置
    frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY
    trigger_type: RebalanceTrigger = RebalanceTrigger.SCHEDULED
    
    # 阈值设置（用于threshold或hybrid触发）
    drift_threshold: float = 0.05    # 权重漂移阈值（5%）
    min_trade_size: float = 0.01     # 最小交易权重（1%）
    
    # 成本设置
    transaction_cost: float = 0.001  # 交易成本（0.1%）
    consider_cost: bool = True       # 是否考虑交易成本
    
    # 时间设置
    rebalance_time: str = "close"    # 再平衡时间点（open/close）
    trading_day_of_week: int = 0     # 每周交易日（0=周一）
    trading_day_of_month: int = 1    # 每月交易日
    
    def __post_init__(self):
        if not 0 <= self.drift_threshold <= 1:
            raise ValueError("drift_threshold必须在[0,1]范围内")


class Rebalancer:
    """
    投资组合再平衡器
    
    管理投资组合的再平衡决策和执行。
    """
    
    def __init__(self, config: RebalanceConfig = None):
        """
        初始化再平衡器
        
        Args:
            config: 再平衡配置
        """
        self.config = config or RebalanceConfig()
        self.last_rebalance_date: Optional[datetime] = None
        logger.info(f"再平衡器初始化完成: 频率={self.config.frequency.value}")
    
    def should_rebalance(self,
                        current_date: datetime,
                        current_weights: pd.Series,
                        target_weights: pd.Series) -> bool:
        """
        判断是否需要再平衡
        
        Args:
            current_date: 当前日期
            current_weights: 当前权重
            target_weights: 目标权重
            
        Returns:
            bool: 是否需要再平衡
        """
        if self.config.trigger_type == RebalanceTrigger.SCHEDULED:
            return self._check_scheduled_trigger(current_date)
        
        elif self.config.trigger_type == RebalanceTrigger.THRESHOLD:
            return self._check_threshold_trigger(current_weights, target_weights)
        
        elif self.config.trigger_type == RebalanceTrigger.HYBRID:
            # 先检查定期，再检查阈值
            if self._check_scheduled_trigger(current_date):
                return True
            return self._check_threshold_trigger(current_weights, target_weights)
        
        return False
    
    def _check_scheduled_trigger(self, current_date: datetime) -> bool:
        """检查定期触发条件"""
        # 如果从未再平衡，则触发
        if self.last_rebalance_date is None:
            return True
        
        # 根据频率检查
        if self.config.frequency == RebalanceFrequency.DAILY:
            return True
        
        elif self.config.frequency == RebalanceFrequency.WEEKLY:
            # 检查是否是指定的每周交易日
            days_since = (current_date - self.last_rebalance_date).days
            is_target_weekday = current_date.weekday() == self.config.trading_day_of_week
            return days_since >= 7 and is_target_weekday
        
        elif self.config.frequency == RebalanceFrequency.BIWEEKLY:
            days_since = (current_date - self.last_rebalance_date).days
            is_target_weekday = current_date.weekday() == self.config.trading_day_of_week
            return days_since >= 14 and is_target_weekday
        
        elif self.config.frequency == RebalanceFrequency.MONTHLY:
            # 检查是否是指定的每月交易日
            is_new_month = current_date.month != self.last_rebalance_date.month
            is_target_day = current_date.day >= self.config.trading_day_of_month
            return is_new_month and is_target_day
        
        elif self.config.frequency == RebalanceFrequency.QUARTERLY:
            months_diff = (current_date.year - self.last_rebalance_date.year) * 12 + \
                         (current_date.month - self.last_rebalance_date.month)
            return months_diff >= 3
        
        return False
    
    def _check_threshold_trigger(self,
                                 current_weights: pd.Series,
                                 target_weights: pd.Series) -> bool:
        """检查阈值触发条件"""
        # 计算权重漂移
        drift = self._calculate_drift(current_weights, target_weights)
        
        # 判断是否超过阈值
        should_trigger = drift > self.config.drift_threshold
        
        if should_trigger:
            logger.info(f"权重漂移 {drift:.4f} 超过阈值 {self.config.drift_threshold:.4f}，触发再平衡")
        
        return should_trigger
    
    def _calculate_drift(self,
                        current_weights: pd.Series,
                        target_weights: pd.Series) -> float:
        """
        计算权重漂移
        
        使用L1范数度量
        """
        # 对齐权重
        all_symbols = set(current_weights.index) | set(target_weights.index)
        aligned_current = pd.Series(0, index=all_symbols)
        aligned_target = pd.Series(0, index=all_symbols)
        
        aligned_current.update(current_weights)
        aligned_target.update(target_weights)
        
        # 计算L1距离
        drift = np.sum(np.abs(aligned_current - aligned_target))
        
        return drift
    
    def calculate_trades(self,
                        current_weights: pd.Series,
                        target_weights: pd.Series,
                        portfolio_value: float = 1000000.0) -> pd.DataFrame:
        """
        计算再平衡所需的交易
        
        Args:
            current_weights: 当前权重
            target_weights: 目标权重
            portfolio_value: 组合总价值
            
        Returns:
            pd.DataFrame: 交易明细
        """
        # 对齐权重
        all_symbols = sorted(set(current_weights.index) | set(target_weights.index))
        aligned_current = pd.Series(0, index=all_symbols)
        aligned_target = pd.Series(0, index=all_symbols)
        
        aligned_current.update(current_weights)
        aligned_target.update(target_weights)
        
        # 计算权重变化
        weight_changes = aligned_target - aligned_current
        
        # 过滤掉小额交易
        significant_changes = weight_changes[
            np.abs(weight_changes) >= self.config.min_trade_size
        ]
        
        if len(significant_changes) == 0:
            logger.info("没有需要执行的交易")
            return pd.DataFrame()
        
        # 构建交易明细
        trades = []
        for symbol in significant_changes.index:
            weight_change = significant_changes[symbol]
            value_change = weight_change * portfolio_value
            
            trade = {
                'symbol': symbol,
                'current_weight': aligned_current[symbol],
                'target_weight': aligned_target[symbol],
                'weight_change': weight_change,
                'value_change': value_change,
                'side': 'BUY' if weight_change > 0 else 'SELL',
                'abs_value': abs(value_change)
            }
            
            # 计算交易成本
            if self.config.consider_cost:
                trade['transaction_cost'] = abs(value_change) * self.config.transaction_cost
            
            trades.append(trade)
        
        trades_df = pd.DataFrame(trades)
        
        # 按交易金额排序
        trades_df = trades_df.sort_values('abs_value', ascending=False)
        
        logger.info(f"生成{len(trades_df)}笔交易，总交易金额: {trades_df['abs_value'].sum():,.2f}")
        
        return trades_df
    
    def optimize_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        优化交易列表
        
        尝试减少交易成本和市场冲击
        """
        if len(trades_df) == 0:
            return trades_df
        
        optimized = trades_df.copy()
        
        # 1. 净额结算（如果有对冲交易）
        # 在实际应用中，这一步可能不需要，因为我们直接计算的是净变化
        
        # 2. 按流动性排序（如果有流动性数据）
        # 优先交易流动性好的股票
        
        # 3. 分批执行大额订单
        # 将大额订单拆分成多个小订单
        
        return optimized
    
    def execute_rebalance(self,
                         current_date: datetime,
                         current_weights: pd.Series,
                         target_weights: pd.Series,
                         portfolio_value: float = 1000000.0) -> Tuple[pd.DataFrame, Dict]:
        """
        执行再平衡
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (交易明细, 再平衡统计)
        """
        # 检查是否需要再平衡
        if not self.should_rebalance(current_date, current_weights, target_weights):
            logger.info(f"{current_date}: 无需再平衡")
            return pd.DataFrame(), {}
        
        # 计算交易
        trades = self.calculate_trades(current_weights, target_weights, portfolio_value)
        
        # 优化交易
        trades = self.optimize_trades(trades)
        
        # 更新最后再平衡日期
        self.last_rebalance_date = current_date
        
        # 统计信息
        stats = {
            'rebalance_date': current_date,
            'n_trades': len(trades),
            'total_turnover': trades['abs_value'].sum() if len(trades) > 0 else 0,
            'total_cost': trades['transaction_cost'].sum() if 'transaction_cost' in trades.columns else 0,
            'drift_before': self._calculate_drift(current_weights, target_weights),
        }
        
        logger.info(f"再平衡完成: {stats['n_trades']}笔交易, 换手率: {stats['total_turnover']:,.2f}")
        
        return trades, stats
    
    def get_rebalance_schedule(self,
                              start_date: datetime,
                              end_date: datetime) -> List[datetime]:
        """
        获取再平衡时间表
        
        返回在指定日期范围内的所有预定再平衡日期
        """
        schedule = []
        current = start_date
        
        while current <= end_date:
            if self.config.frequency == RebalanceFrequency.DAILY:
                schedule.append(current)
                current += timedelta(days=1)
            
            elif self.config.frequency == RebalanceFrequency.WEEKLY:
                if current.weekday() == self.config.trading_day_of_week:
                    schedule.append(current)
                current += timedelta(days=1)
            
            elif self.config.frequency == RebalanceFrequency.MONTHLY:
                if current.day == self.config.trading_day_of_month:
                    schedule.append(current)
                    # 跳到下个月
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)
                else:
                    current += timedelta(days=1)
            
            else:
                break
        
        return schedule

