"""
信号转权重转换器

实现信号到投资组合权重的转换逻辑，支持：
- 横截面分位数选股（long_q/short_q）
- 信号强度归一化
- 多种权重分配方法
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WeightingMethod(Enum):
    """权重分配方法"""
    EQUAL = "equal"                    # 等权重
    SIGNAL_PROPORTIONAL = "signal_proportional"  # 信号比例
    SIGNAL_RANKED = "signal_ranked"    # 信号排名
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # 波动率调整
    SHARPE_WEIGHTED = "sharpe_weighted"  # 夏普比率加权


@dataclass
class SignalTransformConfig:
    """信号转换配置"""
    # 分位数选股参数
    long_quantile: float = 0.8         # 多头信号分位数（选择信号值>80%分位的股票）
    short_quantile: float = 0.2        # 空头信号分位数（选择信号值<20%分位的股票）
    
    # 权重分配方法
    weighting_method: WeightingMethod = WeightingMethod.SIGNAL_PROPORTIONAL
    
    # 权重约束
    max_position_weight: float = 0.05  # 单股最大权重（5%）
    min_position_weight: float = 0.01  # 单股最小权重（1%）
    
    # 归一化参数
    normalize_signals: bool = True     # 是否归一化信号
    signal_clip_std: float = 3.0       # 信号截断标准差（去除极端值）
    
    # 多空配置
    enable_short: bool = False         # 是否允许做空
    long_short_ratio: float = 1.0      # 多空比例（多头资金/空头资金）
    
    # 最小持仓数量
    min_long_positions: int = 5        # 最小多头持仓数量
    max_long_positions: int = 50       # 最大多头持仓数量
    
    def __post_init__(self):
        """配置验证"""
        if not 0 < self.long_quantile <= 1:
            raise ValueError("long_quantile必须在(0,1]范围内")
        if not 0 <= self.short_quantile < 1:
            raise ValueError("short_quantile必须在[0,1)范围内")
        if self.long_quantile <= self.short_quantile:
            raise ValueError("long_quantile必须大于short_quantile")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight必须在(0,1]范围内")


class SignalToWeightTransformer:
    """
    信号转权重转换器
    
    将模型预测信号转换为投资组合权重，支持多种转换方法和约束条件。
    """
    
    def __init__(self, config: SignalTransformConfig = None):
        """
        初始化信号转权重转换器
        
        Args:
            config: 转换配置
        """
        self.config = config or SignalTransformConfig()
        logger.info("信号转权重转换器初始化完成")
    
    def transform(self, 
                  signals: Union[pd.Series, pd.DataFrame],
                  signal_date: Optional[pd.Timestamp] = None,
                  volatility: Optional[pd.Series] = None,
                  sharpe_ratios: Optional[pd.Series] = None) -> pd.Series:
        """
        将信号转换为权重
        
        Args:
            signals: 股票信号（Series或DataFrame的某一列）
            signal_date: 信号日期（用于日志）
            volatility: 股票波动率（用于波动率调整权重）
            sharpe_ratios: 股票夏普比率（用于夏普加权）
            
        Returns:
            pd.Series: 股票权重（index为股票代码，value为权重）
        """
        # 转换为Series
        if isinstance(signals, pd.DataFrame):
            if signal_date and signal_date in signals.index:
                signals = signals.loc[signal_date]
            else:
                signals = signals.iloc[-1]  # 取最后一行
        
        # 去除NaN值
        signals = signals.dropna()
        
        if len(signals) == 0:
            logger.warning("没有有效信号，返回空权重")
            return pd.Series(dtype=float)
        
        # 归一化信号
        if self.config.normalize_signals:
            signals = self._normalize_signals(signals)
        
        # 横截面分位数选股
        selected_stocks = self._select_stocks_by_quantile(signals)
        
        if len(selected_stocks) == 0:
            logger.warning("分位数筛选后没有股票，返回空权重")
            return pd.Series(dtype=float)
        
        # 根据方法计算权重
        if self.config.weighting_method == WeightingMethod.EQUAL:
            weights = self._equal_weights(selected_stocks)
        elif self.config.weighting_method == WeightingMethod.SIGNAL_PROPORTIONAL:
            weights = self._signal_proportional_weights(signals, selected_stocks)
        elif self.config.weighting_method == WeightingMethod.SIGNAL_RANKED:
            weights = self._signal_ranked_weights(signals, selected_stocks)
        elif self.config.weighting_method == WeightingMethod.VOLATILITY_ADJUSTED:
            if volatility is None:
                logger.warning("缺少波动率数据，使用等权重")
                weights = self._equal_weights(selected_stocks)
            else:
                weights = self._volatility_adjusted_weights(signals, selected_stocks, volatility)
        elif self.config.weighting_method == WeightingMethod.SHARPE_WEIGHTED:
            if sharpe_ratios is None:
                logger.warning("缺少夏普比率数据，使用等权重")
                weights = self._equal_weights(selected_stocks)
            else:
                weights = self._sharpe_weighted(selected_stocks, sharpe_ratios)
        else:
            logger.warning(f"未知的权重方法: {self.config.weighting_method}，使用等权重")
            weights = self._equal_weights(selected_stocks)
        
        # 应用权重约束
        weights = self._apply_weight_constraints(weights)
        
        # 归一化权重（确保总和为1）
        weights = self._normalize_weights(weights)
        
        logger.info(f"信号转权重完成: {len(weights)}只股票, 总权重={weights.sum():.4f}")
        
        return weights
    
    def _normalize_signals(self, signals: pd.Series) -> pd.Series:
        """
        归一化信号
        
        使用标准化方法，并截断极端值
        """
        # Z-score标准化
        mean = signals.mean()
        std = signals.std()
        
        if std == 0:
            return pd.Series(0, index=signals.index)
        
        normalized = (signals - mean) / std
        
        # 截断极端值
        if self.config.signal_clip_std > 0:
            normalized = normalized.clip(
                lower=-self.config.signal_clip_std,
                upper=self.config.signal_clip_std
            )
        
        return normalized
    
    def _select_stocks_by_quantile(self, signals: pd.Series) -> List[str]:
        """
        横截面分位数选股
        
        选择信号值在指定分位数范围内的股票
        """
        # 计算分位数阈值
        long_threshold = signals.quantile(self.config.long_quantile)
        
        # 选择多头股票（信号值 > long_quantile）
        long_stocks = signals[signals >= long_threshold].index.tolist()
        
        # 限制持仓数量
        if len(long_stocks) > self.config.max_long_positions:
            # 按信号值排序，取前N只
            long_stocks = signals[signals >= long_threshold].nlargest(
                self.config.max_long_positions
            ).index.tolist()
        
        # 确保最小持仓数量
        if len(long_stocks) < self.config.min_long_positions:
            # 放宽阈值，取信号值最高的min_long_positions只股票
            long_stocks = signals.nlargest(self.config.min_long_positions).index.tolist()
        
        selected_stocks = long_stocks
        
        # 空头选股（如果启用）
        if self.config.enable_short:
            short_threshold = signals.quantile(self.config.short_quantile)
            short_stocks = signals[signals <= short_threshold].index.tolist()
            selected_stocks.extend(short_stocks)
        
        logger.debug(f"分位数选股: 多头{len(long_stocks)}只, 阈值={long_threshold:.4f}")
        
        return selected_stocks
    
    def _equal_weights(self, selected_stocks: List[str]) -> pd.Series:
        """等权重分配"""
        n_stocks = len(selected_stocks)
        weight = 1.0 / n_stocks
        return pd.Series(weight, index=selected_stocks)
    
    def _signal_proportional_weights(self, 
                                     signals: pd.Series, 
                                     selected_stocks: List[str]) -> pd.Series:
        """
        信号比例权重
        
        权重与信号强度成正比
        """
        selected_signals = signals.loc[selected_stocks]
        
        # 确保信号为正数（用于权重分配）
        min_signal = selected_signals.min()
        if min_signal < 0:
            # 平移到正数区域
            selected_signals = selected_signals - min_signal + 1e-6
        
        # 按信号比例分配权重
        weights = selected_signals / selected_signals.sum()
        
        return weights
    
    def _signal_ranked_weights(self, 
                               signals: pd.Series, 
                               selected_stocks: List[str]) -> pd.Series:
        """
        信号排名权重
        
        基于信号排名分配权重，排名越高权重越大
        """
        selected_signals = signals.loc[selected_stocks]
        
        # 计算排名（信号值越大排名越高）
        ranks = selected_signals.rank(ascending=True)
        
        # 按排名分配权重
        weights = ranks / ranks.sum()
        
        return weights
    
    def _volatility_adjusted_weights(self, 
                                     signals: pd.Series,
                                     selected_stocks: List[str],
                                     volatility: pd.Series) -> pd.Series:
        """
        波动率调整权重
        
        高信号、低波动率的股票获得更高权重
        """
        selected_signals = signals.loc[selected_stocks]
        selected_vol = volatility.loc[selected_stocks]
        
        # 风险调整信号 = 信号 / 波动率
        risk_adjusted = selected_signals / (selected_vol + 1e-6)
        
        # 确保为正数
        min_adj = risk_adjusted.min()
        if min_adj < 0:
            risk_adjusted = risk_adjusted - min_adj + 1e-6
        
        # 归一化为权重
        weights = risk_adjusted / risk_adjusted.sum()
        
        return weights
    
    def _sharpe_weighted(self, 
                        selected_stocks: List[str],
                        sharpe_ratios: pd.Series) -> pd.Series:
        """
        夏普比率加权
        
        夏普比率越高权重越大
        """
        selected_sharpe = sharpe_ratios.loc[selected_stocks]
        
        # 确保为正数
        min_sharpe = selected_sharpe.min()
        if min_sharpe < 0:
            selected_sharpe = selected_sharpe - min_sharpe + 1e-6
        
        # 归一化为权重
        weights = selected_sharpe / selected_sharpe.sum()
        
        return weights
    
    def _apply_weight_constraints(self, weights: pd.Series) -> pd.Series:
        """
        应用权重约束
        
        确保单股权重在[min, max]范围内
        """
        # 应用上限
        weights = weights.clip(upper=self.config.max_position_weight)
        
        # 应用下限（去除过小的权重）
        weights = weights[weights >= self.config.min_position_weight]
        
        return weights
    
    def _normalize_weights(self, weights: pd.Series) -> pd.Series:
        """
        归一化权重
        
        确保总权重为1
        """
        total_weight = weights.sum()
        
        if total_weight == 0:
            logger.warning("总权重为0，无法归一化")
            return weights
        
        return weights / total_weight
    
    def batch_transform(self, 
                       signals_df: pd.DataFrame,
                       volatility_df: Optional[pd.DataFrame] = None,
                       sharpe_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        批量转换多个时间点的信号
        
        Args:
            signals_df: 信号DataFrame (日期 × 股票代码)
            volatility_df: 波动率DataFrame (日期 × 股票代码)
            sharpe_df: 夏普比率DataFrame (日期 × 股票代码)
            
        Returns:
            pd.DataFrame: 权重DataFrame (日期 × 股票代码)
        """
        weights_list = []
        dates = []
        
        for date in signals_df.index:
            signals = signals_df.loc[date]
            vol = volatility_df.loc[date] if volatility_df is not None else None
            sharpe = sharpe_df.loc[date] if sharpe_df is not None else None
            
            weights = self.transform(signals, signal_date=date, volatility=vol, sharpe_ratios=sharpe)
            
            if len(weights) > 0:
                weights_list.append(weights)
                dates.append(date)
        
        if len(weights_list) == 0:
            return pd.DataFrame()
        
        # 合并为DataFrame
        weights_df = pd.DataFrame(weights_list, index=dates)
        weights_df = weights_df.fillna(0)
        
        logger.info(f"批量转换完成: {len(dates)}个交易日")
        
        return weights_df

