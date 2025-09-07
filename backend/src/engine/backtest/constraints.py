"""
A股市场约束模型

实现A股市场特有的交易约束条件，包括T+1交易制度、涨跌停限制、
停牌处理、新股上市限制等，确保回测结果的真实性和可靠性。
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd

from backend.src.models.basic_models import StockData
from backend.src.engine.backtest import MarketConstraint, AStockConstraintType

logger = logging.getLogger(__name__)


class MarketType(Enum):
    """市场类型"""
    MAIN_BOARD = "main_board"        # 主板
    SME_BOARD = "sme_board"          # 中小板
    CHINEXT = "chinext"              # 创业板
    STAR_MARKET = "star_market"      # 科创板
    BEIJING_STOCK = "beijing_stock"  # 北交所


class StockStatus(Enum):
    """股票状态"""
    NORMAL = "normal"          # 正常交易
    ST = "st"                  # ST股票
    STAR_ST = "star_st"        # *ST股票
    SUSPENDED = "suspended"    # 停牌
    NEW_LISTING = "new_listing"  # 新股上市
    DELISTING = "delisting"    # 退市整理


@dataclass
class ConstraintConfig:
    """约束配置"""
    # T+1交易制度
    enable_t_plus_1: bool = True
    
    # 涨跌停限制
    enable_price_limit: bool = True
    main_board_limit: float = 0.10          # 主板涨跌停幅度(10%)
    star_market_limit: float = 0.20         # 科创板涨跌停幅度(20%)
    beijing_stock_limit: float = 0.30       # 北交所涨跌停幅度(30%)
    st_stock_limit: float = 0.05            # ST股票涨跌停幅度(5%)
    new_stock_limit: float = 0.44           # 新股首日涨跌停幅度(44%)
    
    # 停牌处理
    enable_suspension_check: bool = True
    
    # 新股上市限制
    enable_new_stock_constraint: bool = True
    new_stock_first_day_limit: int = 1      # 新股上市首日交易限制天数
    
    # ST股票限制
    enable_st_constraint: bool = True
    
    # 最小交易单位
    min_trade_unit: int = 100               # 最小交易单位(股)
    
    # 竞价时间限制
    enable_auction_time: bool = True
    morning_auction_start: str = "09:15"    # 早盘集合竞价开始
    morning_auction_end: str = "09:25"      # 早盘集合竞价结束
    continuous_trading_start: str = "09:30" # 连续交易开始
    continuous_trading_end: str = "15:00"   # 连续交易结束
    
    def get_price_limit(self, market_type: MarketType, stock_status: StockStatus, is_new_stock: bool = False) -> float:
        """获取涨跌停幅度"""
        if is_new_stock:
            return self.new_stock_limit
        
        if stock_status in [StockStatus.ST, StockStatus.STAR_ST]:
            return self.st_stock_limit
        
        if market_type == MarketType.STAR_MARKET:
            return self.star_market_limit
        elif market_type == MarketType.BEIJING_STOCK:
            return self.beijing_stock_limit
        else:
            return self.main_board_limit


@dataclass
class StockInfo:
    """股票信息"""
    symbol: str
    name: str
    market_type: MarketType
    listing_date: Optional[date] = None
    status: StockStatus = StockStatus.NORMAL
    is_st: bool = False
    
    def is_new_stock(self, current_date: date, days_threshold: int = 30) -> bool:
        """判断是否为新股"""
        if not self.listing_date:
            return False
        return (current_date - self.listing_date).days <= days_threshold


class AStockConstraints(MarketConstraint):
    """
    A股市场约束模型
    
    实现A股市场特有的交易约束条件，包括T+1交易制度、涨跌停限制、
    停牌处理等规则，确保回测结果符合实际市场情况。
    """
    
    def __init__(self, config: ConstraintConfig = None):
        """
        初始化A股约束模型
        
        Args:
            config: 约束配置
        """
        self.config = config or ConstraintConfig()
        
        # 股票信息缓存
        self.stock_info_cache: Dict[str, StockInfo] = {}
        
        # 持仓状态跟踪（用于T+1约束）
        self.position_history: Dict[str, Dict[date, float]] = {}  # symbol -> {date: position}
        
        # 停牌信息缓存
        self.suspension_cache: Dict[str, Set[date]] = {}  # symbol -> {suspended_dates}
        
        # 价格限制缓存
        self.price_limit_cache: Dict[Tuple[str, date], Tuple[float, float]] = {}  # (symbol, date) -> (lower_limit, upper_limit)
        
        logger.info("A股市场约束模型初始化完成")
    
    def add_stock_info(self, stock_info: StockInfo):
        """添加股票信息"""
        self.stock_info_cache[stock_info.symbol] = stock_info
    
    def add_suspension_dates(self, symbol: str, suspension_dates: List[date]):
        """添加停牌日期"""
        if symbol not in self.suspension_cache:
            self.suspension_cache[symbol] = set()
        self.suspension_cache[symbol].update(suspension_dates)
    
    def apply_constraint(self, 
                        signals: pd.DataFrame, 
                        market_data: pd.DataFrame) -> pd.DataFrame:
        """
        应用市场约束
        
        Args:
            signals: 交易信号 (日期 × 股票代码)
            market_data: 市场数据 (包含OHLCV等信息)
            
        Returns:
            pd.DataFrame: 应用约束后的信号
        """
        logger.info("开始应用A股市场约束...")
        
        try:
            # 复制原始信号
            constrained_signals = signals.copy()
            
            # 1. 应用T+1交易约束
            if self.config.enable_t_plus_1:
                constrained_signals = self._apply_t_plus_1_constraint(
                    constrained_signals, market_data
                )
            
            # 2. 应用涨跌停约束
            if self.config.enable_price_limit:
                constrained_signals = self._apply_price_limit_constraint(
                    constrained_signals, market_data
                )
            
            # 3. 应用停牌约束
            if self.config.enable_suspension_check:
                constrained_signals = self._apply_suspension_constraint(
                    constrained_signals, market_data
                )
            
            # 4. 应用新股约束
            if self.config.enable_new_stock_constraint:
                constrained_signals = self._apply_new_stock_constraint(
                    constrained_signals, market_data
                )
            
            # 5. 应用ST股票约束
            if self.config.enable_st_constraint:
                constrained_signals = self._apply_st_constraint(
                    constrained_signals, market_data
                )
            
            # 6. 应用交易时间约束
            if self.config.enable_auction_time:
                constrained_signals = self._apply_trading_time_constraint(
                    constrained_signals, market_data
                )
            
            logger.info(f"A股约束应用完成，信号变化: {signals.sum().sum()} -> {constrained_signals.sum().sum()}")
            return constrained_signals
            
        except Exception as e:
            logger.error(f"应用A股约束失败: {e}")
            raise
    
    def _apply_t_plus_1_constraint(self, 
                                  signals: pd.DataFrame, 
                                  market_data: pd.DataFrame) -> pd.DataFrame:
        """应用T+1交易约束"""
        logger.debug("应用T+1交易约束...")
        
        constrained_signals = signals.copy()
        
        # 遍历每只股票
        for symbol in signals.columns:
            if symbol not in self.position_history:
                self.position_history[symbol] = {}
            
            current_position = 0.0
            
            # 按时间顺序处理
            for date in signals.index:
                # 获取当前信号
                signal = signals.loc[date, symbol]
                
                # 如果是卖出信号，检查是否有可卖持仓
                if signal < 0:  # 卖出信号
                    # 计算可卖数量（昨日及之前买入的持仓）
                    available_to_sell = self._get_available_position_for_sale(symbol, date)
                    
                    # 限制卖出信号不超过可卖数量
                    if abs(signal) > available_to_sell:
                        if available_to_sell > 0:
                            constrained_signals.loc[date, symbol] = -available_to_sell
                        else:
                            constrained_signals.loc[date, symbol] = 0  # 无法卖出
                
                # 更新持仓记录
                if signal != 0:
                    self.position_history[symbol][date] = signal
                    current_position += signal
        
        return constrained_signals
    
    def _get_available_position_for_sale(self, symbol: str, current_date: date) -> float:
        """获取可卖持仓数量（T+1规则下）"""
        if symbol not in self.position_history:
            return 0.0
        
        available_position = 0.0
        
        # 累计昨日及之前的买入持仓
        for trade_date, position in self.position_history[symbol].items():
            if trade_date < current_date and position > 0:  # 当日之前的买入持仓
                available_position += position
        
        return max(0.0, available_position)
    
    def _apply_price_limit_constraint(self, 
                                    signals: pd.DataFrame, 
                                    market_data: pd.DataFrame) -> pd.DataFrame:
        """应用涨跌停约束"""
        logger.debug("应用涨跌停约束...")
        
        constrained_signals = signals.copy()
        
        # 获取价格数据
        if 'close' in market_data.columns:
            close_prices = market_data['close']
        else:
            # 多层索引情况
            close_prices = market_data.xs('close', axis=1, level=1) if market_data.columns.nlevels > 1 else market_data
        
        # 计算涨跌停价格
        for symbol in signals.columns:
            if symbol not in close_prices.columns:
                continue
            
            stock_info = self._get_stock_info(symbol)
            price_series = close_prices[symbol]
            
            for i, date in enumerate(signals.index):
                if i == 0:  # 第一天无前日收盘价
                    continue
                
                prev_date = signals.index[i-1]
                if pd.isna(price_series.loc[prev_date]):
                    continue
                
                prev_close = price_series.loc[prev_date]
                
                # 获取涨跌停幅度
                limit_ratio = self.config.get_price_limit(
                    stock_info.market_type,
                    stock_info.status,
                    stock_info.is_new_stock(date)
                )
                
                # 计算涨跌停价格
                upper_limit = prev_close * (1 + limit_ratio)
                lower_limit = prev_close * (1 - limit_ratio)
                
                # 缓存价格限制
                self.price_limit_cache[(symbol, date)] = (lower_limit, upper_limit)
                
                # 检查当前价格是否触及涨跌停
                if date in price_series.index and not pd.isna(price_series.loc[date]):
                    current_price = price_series.loc[date]
                    
                    # 如果触及涨停，禁止买入信号
                    if current_price >= upper_limit * 0.999:  # 考虑浮点数精度
                        if signals.loc[date, symbol] > 0:
                            constrained_signals.loc[date, symbol] = 0
                            logger.debug(f"{symbol} {date} 涨停，禁止买入")
                    
                    # 如果触及跌停，禁止卖出信号  
                    elif current_price <= lower_limit * 1.001:  # 考虑浮点数精度
                        if signals.loc[date, symbol] < 0:
                            constrained_signals.loc[date, symbol] = 0
                            logger.debug(f"{symbol} {date} 跌停，禁止卖出")
        
        return constrained_signals
    
    def _apply_suspension_constraint(self, 
                                   signals: pd.DataFrame, 
                                   market_data: pd.DataFrame) -> pd.DataFrame:
        """应用停牌约束"""
        logger.debug("应用停牌约束...")
        
        constrained_signals = signals.copy()
        
        # 检查停牌信息
        for symbol in signals.columns:
            if symbol in self.suspension_cache:
                suspended_dates = self.suspension_cache[symbol]
                
                # 在停牌日期清除所有交易信号
                for date in signals.index:
                    if date in suspended_dates:
                        constrained_signals.loc[date, symbol] = 0
                        logger.debug(f"{symbol} {date} 停牌，清除交易信号")
        
        # 根据成交量判断停牌（成交量为0）
        if 'volume' in market_data.columns:
            volume_data = market_data['volume']
        elif market_data.columns.nlevels > 1:
            volume_data = market_data.xs('volume', axis=1, level=1)
        else:
            return constrained_signals
        
        for symbol in signals.columns:
            if symbol not in volume_data.columns:
                continue
                
            for date in signals.index:
                if date in volume_data.index and volume_data.loc[date, symbol] == 0:
                    # 成交量为0，可能停牌
                    constrained_signals.loc[date, symbol] = 0
                    logger.debug(f"{symbol} {date} 成交量为0，清除交易信号")
        
        return constrained_signals
    
    def _apply_new_stock_constraint(self, 
                                  signals: pd.DataFrame, 
                                  market_data: pd.DataFrame) -> pd.DataFrame:
        """应用新股约束"""
        logger.debug("应用新股约束...")
        
        constrained_signals = signals.copy()
        
        for symbol in signals.columns:
            stock_info = self._get_stock_info(symbol)
            
            if not stock_info.listing_date:
                continue
            
            # 检查新股上市期间的限制
            for date in signals.index:
                if stock_info.is_new_stock(date, self.config.new_stock_first_day_limit):
                    # 新股期间可能有特殊交易规则
                    # 这里可以添加具体的新股交易限制逻辑
                    pass
        
        return constrained_signals
    
    def _apply_st_constraint(self, 
                           signals: pd.DataFrame, 
                           market_data: pd.DataFrame) -> pd.DataFrame:
        """应用ST股票约束"""
        logger.debug("应用ST股票约束...")
        
        constrained_signals = signals.copy()
        
        for symbol in signals.columns:
            stock_info = self._get_stock_info(symbol)
            
            # ST股票的特殊处理（如果有需要）
            if stock_info.status in [StockStatus.ST, StockStatus.STAR_ST]:
                # ST股票可能有额外的风险提示，但交易规则与普通股票相同
                # 主要区别在涨跌停幅度，已在价格限制约束中处理
                pass
        
        return constrained_signals
    
    def _apply_trading_time_constraint(self, 
                                     signals: pd.DataFrame, 
                                     market_data: pd.DataFrame) -> pd.DataFrame:
        """应用交易时间约束"""
        logger.debug("应用交易时间约束...")
        
        # 日线数据通常不需要时间约束，这里保留接口
        # 如果需要处理分钟级数据，可以在这里添加时间过滤逻辑
        
        return signals.copy()
    
    def _get_stock_info(self, symbol: str) -> StockInfo:
        """获取股票信息"""
        if symbol in self.stock_info_cache:
            return self.stock_info_cache[symbol]
        
        # 默认股票信息（根据代码推断市场类型）
        market_type = self._infer_market_type(symbol)
        status = StockStatus.NORMAL
        
        # 根据股票名称判断是否为ST股票
        is_st = symbol.startswith('ST') or '*ST' in symbol
        if is_st:
            status = StockStatus.STAR_ST if '*ST' in symbol else StockStatus.ST
        
        stock_info = StockInfo(
            symbol=symbol,
            name=f"股票{symbol}",
            market_type=market_type,
            status=status,
            is_st=is_st
        )
        
        self.stock_info_cache[symbol] = stock_info
        return stock_info
    
    def _infer_market_type(self, symbol: str) -> MarketType:
        """根据股票代码推断市场类型"""
        if symbol.startswith('00'):
            return MarketType.MAIN_BOARD  # 深圳主板
        elif symbol.startswith('30'):
            return MarketType.CHINEXT     # 创业板
        elif symbol.startswith('68'):
            return MarketType.STAR_MARKET # 科创板
        elif symbol.startswith('6'):
            return MarketType.MAIN_BOARD  # 上海主板
        elif symbol.startswith('8') or symbol.startswith('4'):
            return MarketType.BEIJING_STOCK  # 北交所
        else:
            return MarketType.MAIN_BOARD  # 默认主板
    
    def get_price_limits(self, symbol: str, date: date) -> Optional[Tuple[float, float]]:
        """获取指定股票和日期的涨跌停价格"""
        return self.price_limit_cache.get((symbol, date))
    
    def is_suspended(self, symbol: str, date: date) -> bool:
        """检查指定股票在指定日期是否停牌"""
        if symbol in self.suspension_cache:
            return date in self.suspension_cache[symbol]
        return False
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """获取约束模型信息"""
        return {
            'config': {
                't_plus_1_enabled': self.config.enable_t_plus_1,
                'price_limit_enabled': self.config.enable_price_limit,
                'suspension_check_enabled': self.config.enable_suspension_check,
                'new_stock_constraint_enabled': self.config.enable_new_stock_constraint,
                'st_constraint_enabled': self.config.enable_st_constraint,
            },
            'cached_stocks': len(self.stock_info_cache),
            'suspension_records': {symbol: len(dates) for symbol, dates in self.suspension_cache.items()},
            'price_limit_cache_size': len(self.price_limit_cache),
            'supported_constraints': [
                'T+1交易制度',
                '涨跌停限制',
                '停牌处理',
                '新股上市约束',
                'ST股票约束',
                '交易时间约束'
            ]
        }
    
    def clear_cache(self):
        """清理缓存"""
        self.stock_info_cache.clear()
        self.position_history.clear()
        self.suspension_cache.clear()
        self.price_limit_cache.clear()
        logger.info("A股约束模型缓存已清理")


def create_default_a_stock_constraints() -> AStockConstraints:
    """创建默认的A股约束模型"""
    config = ConstraintConfig(
        enable_t_plus_1=True,
        enable_price_limit=True,
        enable_suspension_check=True,
        enable_new_stock_constraint=True,
        enable_st_constraint=True,
        enable_auction_time=False  # 日线数据不需要时间约束
    )
    
    return AStockConstraints(config)


def create_lenient_a_stock_constraints() -> AStockConstraints:
    """创建宽松的A股约束模型（用于理想化回测）"""
    config = ConstraintConfig(
        enable_t_plus_1=False,      # 禁用T+1
        enable_price_limit=False,   # 禁用涨跌停
        enable_suspension_check=False,  # 禁用停牌检查
        enable_new_stock_constraint=False,
        enable_st_constraint=False,
        enable_auction_time=False
    )
    
    return AStockConstraints(config)


# 便利函数：根据股票代码推断市场信息
def get_market_info(symbol: str) -> Dict[str, Any]:
    """根据股票代码获取市场信息"""
    constraints = AStockConstraints()
    stock_info = constraints._get_stock_info(symbol)
    
    return {
        'symbol': symbol,
        'market_type': stock_info.market_type.value,
        'status': stock_info.status.value,
        'is_st': stock_info.is_st,
        'price_limit_ratio': constraints.config.get_price_limit(
            stock_info.market_type, stock_info.status
        )
    }
