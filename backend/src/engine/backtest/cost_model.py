"""
交易成本模型

实现完整的A股交易成本计算，包括手续费、印花税、过户费、
滑点成本等，确保回测结果考虑实际交易成本的影响。
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd

from backend.src.engine.backtest import CostModel

logger = logging.getLogger(__name__)


class CostType(Enum):
    """成本类型"""
    COMMISSION = "commission"         # 佣金
    STAMP_TAX = "stamp_tax"          # 印花税
    TRANSFER_FEE = "transfer_fee"    # 过户费
    SLIPPAGE = "slippage"           # 滑点成本
    MARKET_IMPACT = "market_impact" # 市场冲击成本


class BrokerType(Enum):
    """券商类型"""
    DISCOUNT = "discount"     # 低佣券商
    STANDARD = "standard"     # 标准券商
    PREMIUM = "premium"       # 高端券商
    CUSTOM = "custom"         # 自定义费率


@dataclass
class CostConfig:
    """交易成本配置"""
    # 佣金费率配置
    commission_rate: float = 0.0003        # 万三佣金（0.03%）
    commission_min: float = 5.0            # 最低佣金5元
    commission_both_ways: bool = True      # 买卖双向收费
    
    # 印花税配置
    stamp_tax_rate: float = 0.001          # 印花税千分之一（0.1%）
    stamp_tax_sell_only: bool = True       # 仅卖出收印花税
    
    # 过户费配置
    transfer_fee_rate: float = 0.00002     # 过户费万分之0.2（0.002%）
    transfer_fee_min: float = 1.0          # 最低过户费1元
    transfer_fee_both_ways: bool = True    # 买卖双向收费
    transfer_fee_shanghai_only: bool = True # 仅上海股票收过户费（历史规则）
    
    # 滑点成本配置
    base_slippage_rate: float = 0.0005     # 基础滑点万分之五（0.05%）
    volume_impact_factor: float = 0.1      # 成交量冲击系数
    volatility_impact_factor: float = 0.05 # 波动率冲击系数
    liquidity_threshold: float = 1000000   # 流动性阈值（成交量）
    
    # 市场冲击成本配置
    enable_market_impact: bool = True      # 是否启用市场冲击成本
    large_order_threshold: float = 100000  # 大单阈值（金额）
    impact_decay_factor: float = 0.5      # 冲击衰减因子
    
    # 券商类型
    broker_type: BrokerType = BrokerType.STANDARD
    
    def get_effective_commission_rate(self) -> float:
        """获取有效佣金费率"""
        broker_rates = {
            BrokerType.DISCOUNT: 0.0003,   # 万三
            BrokerType.STANDARD: 0.0005,   # 万五
            BrokerType.PREMIUM: 0.0008,    # 万八
            BrokerType.CUSTOM: self.commission_rate
        }
        return broker_rates.get(self.broker_type, self.commission_rate)


@dataclass
class TradeInfo:
    """交易信息"""
    symbol: str                    # 股票代码
    side: str                     # 买卖方向 ('buy', 'sell')
    quantity: float               # 交易数量（股）
    price: float                  # 交易价格
    amount: float                 # 交易金额
    timestamp: datetime           # 交易时间
    
    # 市场状态
    avg_volume: Optional[float] = None      # 平均成交量
    volatility: Optional[float] = None      # 波动率
    bid_ask_spread: Optional[float] = None  # 买卖价差
    market_cap: Optional[float] = None      # 市值
    
    def __post_init__(self):
        if self.amount == 0:
            self.amount = self.quantity * self.price


@dataclass
class CostBreakdown:
    """成本分解"""
    commission: float = 0.0        # 佣金
    stamp_tax: float = 0.0         # 印花税
    transfer_fee: float = 0.0      # 过户费
    slippage: float = 0.0          # 滑点成本
    market_impact: float = 0.0     # 市场冲击成本
    total_cost: float = 0.0        # 总成本
    cost_rate: float = 0.0         # 成本率（成本/交易金额）
    
    def calculate_total(self):
        """计算总成本"""
        self.total_cost = (
            self.commission + 
            self.stamp_tax + 
            self.transfer_fee + 
            self.slippage + 
            self.market_impact
        )
        
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'commission': self.commission,
            'stamp_tax': self.stamp_tax,
            'transfer_fee': self.transfer_fee,
            'slippage': self.slippage,
            'market_impact': self.market_impact,
            'total_cost': self.total_cost,
            'cost_rate': self.cost_rate
        }


class TradingCostModel(CostModel):
    """
    交易成本模型
    
    实现完整的A股交易成本计算，包括佣金、印花税、过户费、
    滑点成本等，支持不同券商费率和市场冲击建模。
    """
    
    def __init__(self, config: CostConfig = None):
        """
        初始化交易成本模型
        
        Args:
            config: 成本配置
        """
        self.config = config or CostConfig()
        
        # 成本历史记录
        self.cost_history: List[CostBreakdown] = []
        self.trade_history: List[TradeInfo] = []
        
        # 统计信息
        self.total_trades = 0
        self.total_cost = 0.0
        self.total_amount = 0.0
        
        logger.info(f"交易成本模型初始化完成 - 券商类型: {self.config.broker_type.value}")
    
    def calculate_cost(self, 
                      trades: Union[TradeInfo, List[TradeInfo], pd.DataFrame], 
                      market_data: Optional[pd.DataFrame] = None) -> Union[CostBreakdown, List[CostBreakdown], pd.DataFrame]:
        """
        计算交易成本
        
        Args:
            trades: 交易信息
            market_data: 市场数据（用于滑点计算）
            
        Returns:
            Union[CostBreakdown, List[CostBreakdown], pd.DataFrame]: 成本分解结果
        """
        if isinstance(trades, TradeInfo):
            return self._calculate_single_trade_cost(trades, market_data)
        elif isinstance(trades, list):
            return [self._calculate_single_trade_cost(trade, market_data) for trade in trades]
        elif isinstance(trades, pd.DataFrame):
            return self._calculate_dataframe_costs(trades, market_data)
        else:
            raise ValueError("不支持的交易数据格式")
    
    def _calculate_single_trade_cost(self, 
                                   trade: TradeInfo, 
                                   market_data: Optional[pd.DataFrame] = None) -> CostBreakdown:
        """计算单笔交易成本"""
        try:
            cost_breakdown = CostBreakdown()
            
            # 1. 佣金计算
            cost_breakdown.commission = self._calculate_commission(trade)
            
            # 2. 印花税计算
            cost_breakdown.stamp_tax = self._calculate_stamp_tax(trade)
            
            # 3. 过户费计算
            cost_breakdown.transfer_fee = self._calculate_transfer_fee(trade)
            
            # 4. 滑点成本计算
            cost_breakdown.slippage = self._calculate_slippage_cost(trade, market_data)
            
            # 5. 市场冲击成本计算
            if self.config.enable_market_impact:
                cost_breakdown.market_impact = self._calculate_market_impact_cost(trade, market_data)
            
            # 6. 计算总成本和成本率
            cost_breakdown.calculate_total()
            if trade.amount > 0:
                cost_breakdown.cost_rate = cost_breakdown.total_cost / trade.amount
            
            # 7. 记录成本历史
            self.cost_history.append(cost_breakdown)
            self.trade_history.append(trade)
            self.total_trades += 1
            self.total_cost += cost_breakdown.total_cost
            self.total_amount += trade.amount
            
            return cost_breakdown
            
        except Exception as e:
            logger.error(f"计算交易成本失败: {e}")
            # 返回零成本作为兜底
            return CostBreakdown()
    
    def _calculate_commission(self, trade: TradeInfo) -> float:
        """计算佣金"""
        if not self.config.commission_both_ways and trade.side == 'sell':
            return 0.0
        
        commission_rate = self.config.get_effective_commission_rate()
        commission = trade.amount * commission_rate
        
        # 应用最低佣金
        commission = max(commission, self.config.commission_min)
        
        return commission
    
    def _calculate_stamp_tax(self, trade: TradeInfo) -> float:
        """计算印花税"""
        # 印花税仅在卖出时收取
        if self.config.stamp_tax_sell_only and trade.side != 'sell':
            return 0.0
        
        return trade.amount * self.config.stamp_tax_rate
    
    def _calculate_transfer_fee(self, trade: TradeInfo) -> float:
        """计算过户费"""
        # 历史上仅上海股票收过户费，现在已改为全市场
        if self.config.transfer_fee_shanghai_only and not trade.symbol.startswith('6'):
            return 0.0
        
        if not self.config.transfer_fee_both_ways and trade.side == 'sell':
            return 0.0
        
        transfer_fee = trade.amount * self.config.transfer_fee_rate
        
        # 应用最低过户费
        transfer_fee = max(transfer_fee, self.config.transfer_fee_min)
        
        return transfer_fee
    
    def _calculate_slippage_cost(self, 
                               trade: TradeInfo, 
                               market_data: Optional[pd.DataFrame] = None) -> float:
        """计算滑点成本"""
        try:
            base_slippage = trade.amount * self.config.base_slippage_rate
            
            # 如果没有市场数据，返回基础滑点
            if market_data is None or trade.avg_volume is None:
                return base_slippage
            
            # 根据交易量调整滑点
            volume_impact = 0.0
            if trade.avg_volume > 0:
                volume_ratio = trade.quantity / trade.avg_volume
                volume_impact = base_slippage * volume_ratio * self.config.volume_impact_factor
            
            # 根据波动率调整滑点
            volatility_impact = 0.0
            if trade.volatility is not None and trade.volatility > 0:
                volatility_impact = base_slippage * trade.volatility * self.config.volatility_impact_factor
            
            # 流动性调整
            liquidity_factor = 1.0
            if trade.avg_volume is not None and trade.avg_volume < self.config.liquidity_threshold:
                liquidity_factor = 1.5  # 低流动性增加滑点
            
            total_slippage = (base_slippage + volume_impact + volatility_impact) * liquidity_factor
            
            return max(total_slippage, 0.0)
            
        except Exception as e:
            logger.warning(f"计算滑点成本失败，使用基础滑点: {e}")
            return trade.amount * self.config.base_slippage_rate
    
    def _calculate_market_impact_cost(self, 
                                    trade: TradeInfo, 
                                    market_data: Optional[pd.DataFrame] = None) -> float:
        """计算市场冲击成本"""
        try:
            # 只对大单计算市场冲击
            if trade.amount < self.config.large_order_threshold:
                return 0.0
            
            # 基础冲击成本
            impact_ratio = (trade.amount - self.config.large_order_threshold) / self.config.large_order_threshold
            base_impact = trade.amount * 0.0001 * impact_ratio  # 万分之一的基础冲击
            
            # 根据买卖价差调整
            spread_impact = 0.0
            if trade.bid_ask_spread is not None:
                spread_impact = trade.amount * trade.bid_ask_spread * 0.5  # 价差的一半作为冲击
            
            # 市值调整（小盘股冲击更大）
            market_cap_factor = 1.0
            if trade.market_cap is not None and trade.market_cap < 10000000000:  # 100亿市值以下
                market_cap_factor = 1.2
            
            total_impact = (base_impact + spread_impact) * market_cap_factor
            
            return max(total_impact, 0.0)
            
        except Exception as e:
            logger.warning(f"计算市场冲击成本失败: {e}")
            return 0.0
    
    def _calculate_dataframe_costs(self, 
                                 trades_df: pd.DataFrame, 
                                 market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """批量计算DataFrame格式的交易成本"""
        results = []
        
        for _, row in trades_df.iterrows():
            # 构造TradeInfo对象
            trade = TradeInfo(
                symbol=row.get('symbol', ''),
                side=row.get('side', 'buy'),
                quantity=row.get('quantity', 0),
                price=row.get('price', 0),
                amount=row.get('amount', row.get('quantity', 0) * row.get('price', 0)),
                timestamp=row.get('timestamp', datetime.now()),
                avg_volume=row.get('avg_volume'),
                volatility=row.get('volatility'),
                bid_ask_spread=row.get('bid_ask_spread'),
                market_cap=row.get('market_cap')
            )
            
            # 计算成本
            cost_breakdown = self._calculate_single_trade_cost(trade, market_data)
            
            # 添加到结果
            result_row = row.to_dict()
            result_row.update(cost_breakdown.to_dict())
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def get_cost_breakdown(self) -> Dict[str, str]:
        """获取成本组成说明"""
        return {
            'commission': f'佣金 - {self.config.get_effective_commission_rate():.4f}% (最低{self.config.commission_min}元)',
            'stamp_tax': f'印花税 - {self.config.stamp_tax_rate:.3f}% (仅卖出)' if self.config.stamp_tax_sell_only else f'印花税 - {self.config.stamp_tax_rate:.3f}%',
            'transfer_fee': f'过户费 - {self.config.transfer_fee_rate:.5f}% (最低{self.config.transfer_fee_min}元)',
            'slippage': f'滑点成本 - 基础{self.config.base_slippage_rate:.4f}% + 市场影响',
            'market_impact': f'市场冲击 - 大单(>{self.config.large_order_threshold:,.0f}元)额外成本'
        }
    
    def generate_cost_analysis_report(self) -> Dict[str, Any]:
        """生成成本分析报告"""
        if not self.cost_history:
            return {'error': '暂无交易成本数据'}
        
        # 成本统计
        cost_df = pd.DataFrame([cost.to_dict() for cost in self.cost_history])
        
        report = {
            'summary': {
                'total_trades': self.total_trades,
                'total_cost': self.total_cost,
                'total_amount': self.total_amount,
                'average_cost_rate': self.total_cost / self.total_amount if self.total_amount > 0 else 0,
                'cost_breakdown': self.get_cost_breakdown()
            },
            'statistics': {
                'cost_rate': {
                    'mean': float(cost_df['cost_rate'].mean()),
                    'std': float(cost_df['cost_rate'].std()),
                    'min': float(cost_df['cost_rate'].min()),
                    'max': float(cost_df['cost_rate'].max()),
                    'percentiles': {
                        '25%': float(cost_df['cost_rate'].quantile(0.25)),
                        '50%': float(cost_df['cost_rate'].quantile(0.50)),
                        '75%': float(cost_df['cost_rate'].quantile(0.75)),
                        '95%': float(cost_df['cost_rate'].quantile(0.95))
                    }
                },
                'cost_components': {
                    'commission_pct': float(cost_df['commission'].sum() / cost_df['total_cost'].sum() * 100),
                    'stamp_tax_pct': float(cost_df['stamp_tax'].sum() / cost_df['total_cost'].sum() * 100),
                    'transfer_fee_pct': float(cost_df['transfer_fee'].sum() / cost_df['total_cost'].sum() * 100),
                    'slippage_pct': float(cost_df['slippage'].sum() / cost_df['total_cost'].sum() * 100),
                    'market_impact_pct': float(cost_df['market_impact'].sum() / cost_df['total_cost'].sum() * 100)
                }
            },
            'recommendations': self._generate_cost_recommendations(cost_df)
        }
        
        return report
    
    def _generate_cost_recommendations(self, cost_df: pd.DataFrame) -> List[str]:
        """生成成本优化建议"""
        recommendations = []
        
        avg_cost_rate = cost_df['cost_rate'].mean()
        
        # 成本率分析
        if avg_cost_rate > 0.005:  # 成本率超过0.5%
            recommendations.append("交易成本偏高，建议选择低佣券商或增大单笔交易金额")
        
        # 佣金分析
        commission_pct = cost_df['commission'].sum() / cost_df['total_cost'].sum()
        if commission_pct > 0.4:  # 佣金占比超过40%
            recommendations.append("佣金是主要成本，建议选择万三或万二的低佣券商")
        
        # 滑点分析
        slippage_pct = cost_df['slippage'].sum() / cost_df['total_cost'].sum()
        if slippage_pct > 0.3:  # 滑点占比超过30%
            recommendations.append("滑点成本较高，建议优化交易时机，避免在开盘收盘等波动大的时段交易")
        
        # 交易频率分析
        if len(cost_df) > 1000:  # 交易次数过多
            recommendations.append("交易频率过高，建议降低交易频率以减少总体交易成本")
        
        if not recommendations:
            recommendations.append("交易成本控制良好，继续保持当前交易策略")
        
        return recommendations
    
    def estimate_cost_impact_on_returns(self, 
                                      strategy_returns: pd.Series, 
                                      turnover_rate: float = 2.0) -> Dict[str, float]:
        """
        估算成本对收益的影响
        
        Args:
            strategy_returns: 策略收益率序列
            turnover_rate: 年换手率（默认200%）
            
        Returns:
            Dict[str, float]: 成本影响分析
        """
        # 计算平均成本率
        if not self.cost_history:
            avg_cost_rate = self.config.get_effective_commission_rate() + self.config.stamp_tax_rate * 0.5  # 假设一半卖出
        else:
            avg_cost_rate = sum(cost.cost_rate for cost in self.cost_history) / len(self.cost_history)
        
        # 年化成本
        annual_cost_drag = avg_cost_rate * turnover_rate
        
        # 对收益的影响
        strategy_annual_return = (1 + strategy_returns.mean()) ** 252 - 1
        net_annual_return = strategy_annual_return - annual_cost_drag
        
        return {
            'average_cost_rate': avg_cost_rate,
            'annual_turnover': turnover_rate,
            'annual_cost_drag': annual_cost_drag,
            'gross_annual_return': strategy_annual_return,
            'net_annual_return': net_annual_return,
            'cost_impact_on_return': annual_cost_drag / abs(strategy_annual_return) if strategy_annual_return != 0 else 0,
            'cost_adjusted_sharpe_ratio': self._calculate_cost_adjusted_sharpe(strategy_returns, annual_cost_drag)
        }
    
    def _calculate_cost_adjusted_sharpe(self, returns: pd.Series, annual_cost_drag: float) -> float:
        """计算扣除成本后的夏普比率"""
        try:
            # 日均成本拖累
            daily_cost_drag = annual_cost_drag / 252
            
            # 扣除成本后的收益
            net_returns = returns - daily_cost_drag
            
            # 计算夏普比率
            if net_returns.std() == 0:
                return 0.0
            
            risk_free_rate = 0.03  # 假设3%无风险利率
            excess_returns = net_returns.mean() - risk_free_rate / 252
            
            return excess_returns / net_returns.std() * np.sqrt(252)
            
        except:
            return 0.0
    
    def clear_history(self):
        """清空成本历史"""
        self.cost_history.clear()
        self.trade_history.clear()
        self.total_trades = 0
        self.total_cost = 0.0
        self.total_amount = 0.0
        logger.info("成本模型历史数据已清空")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'broker_type': self.config.broker_type.value,
            'commission_rate': self.config.get_effective_commission_rate(),
            'stamp_tax_rate': self.config.stamp_tax_rate,
            'transfer_fee_rate': self.config.transfer_fee_rate,
            'base_slippage_rate': self.config.base_slippage_rate,
            'enable_market_impact': self.config.enable_market_impact,
            'total_trades': self.total_trades,
            'total_cost': self.total_cost,
            'total_amount': self.total_amount,
            'supported_cost_types': [cost_type.value for cost_type in CostType]
        }


def create_standard_cost_model() -> TradingCostModel:
    """创建标准交易成本模型"""
    config = CostConfig(
        broker_type=BrokerType.STANDARD,
        commission_rate=0.0005,      # 万五佣金
        commission_min=5.0,
        stamp_tax_rate=0.001,        # 千分之一印花税
        transfer_fee_rate=0.00002,   # 万分之0.2过户费
        base_slippage_rate=0.0005,   # 万分之五滑点
        enable_market_impact=True
    )
    
    return TradingCostModel(config)


def create_discount_cost_model() -> TradingCostModel:
    """创建低佣交易成本模型"""
    config = CostConfig(
        broker_type=BrokerType.DISCOUNT,
        commission_rate=0.0003,      # 万三佣金
        commission_min=1.0,          # 低佣券商最低佣金较低
        stamp_tax_rate=0.001,
        transfer_fee_rate=0.00002,
        base_slippage_rate=0.0003,   # 低滑点假设
        enable_market_impact=True
    )
    
    return TradingCostModel(config)


def create_minimal_cost_model() -> TradingCostModel:
    """创建最小成本模型（用于理想化回测）"""
    config = CostConfig(
        commission_rate=0.0001,      # 极低佣金
        commission_min=0.1,
        stamp_tax_rate=0.001,        # 仍保留印花税
        transfer_fee_rate=0.00001,
        base_slippage_rate=0.0001,
        enable_market_impact=False   # 不考虑市场冲击
    )
    
    return TradingCostModel(config)



