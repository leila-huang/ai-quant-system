"""
技术指标计算引擎

实现常用的技术指标计算，包括移动平均线、RSI、MACD、布林带等。
采用向量化计算优化性能，支持批量处理。
"""

from typing import Dict, List, Optional, Union, Tuple
import warnings
import logging
import pandas as pd
import numpy as np
from numba import jit

from backend.src.models.basic_models import StockData
from backend.src.engine.features import TechnicalIndicatorCalculator

# 禁用pandas性能警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class TechnicalIndicators(TechnicalIndicatorCalculator):
    """
    技术指标计算器
    
    提供常用技术指标的高性能向量化计算，支持多种参数配置。
    """
    
    def __init__(self):
        self._supported_indicators = [
            "MA", "EMA", "RSI", "MACD", "BOLL", "KDJ", "WILLIAMS", "VOLUME_MA"
        ]
    
    def get_supported_indicators(self) -> List[str]:
        """获取支持的技术指标列表"""
        return self._supported_indicators.copy()
    
    def calculate(self, data: Union[StockData, pd.DataFrame], 
                 indicators: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 输入数据（StockData对象或DataFrame）
            indicators: 要计算的指标列表，None表示计算所有指标
            **kwargs: 指标参数
            
        Returns:
            pd.DataFrame: 包含所有指标的DataFrame
        """
        # 数据转换和验证
        df = self._prepare_data(data)
        if df.empty:
            return df
            
        # 默认计算所有指标
        if indicators is None:
            indicators = self._supported_indicators
            
        # 验证指标名称
        invalid_indicators = set(indicators) - set(self._supported_indicators)
        if invalid_indicators:
            raise ValueError(f"不支持的技术指标: {invalid_indicators}")
        
        result_df = df.copy()
        
        try:
            # 计算各项技术指标
            for indicator in indicators:
                if indicator == "MA":
                    ma_windows = kwargs.get('ma_windows', [5, 10, 20, 30, 60])
                    result_df = self._add_ma_indicators(result_df, ma_windows=ma_windows)
                elif indicator == "EMA":
                    ema_windows = kwargs.get('ema_windows', [12, 26])
                    result_df = self._add_ema_indicators(result_df, ema_windows=ema_windows)
                elif indicator == "RSI":
                    rsi_windows = kwargs.get('rsi_windows', [14, 6])
                    result_df = self._add_rsi_indicators(result_df, rsi_windows=rsi_windows)
                elif indicator == "MACD":
                    fast_period = kwargs.get('fast_period', 12)
                    slow_period = kwargs.get('slow_period', 26)
                    signal_period = kwargs.get('signal_period', 9)
                    result_df = self._add_macd_indicators(result_df, fast_period, slow_period, signal_period)
                elif indicator == "BOLL":
                    window = kwargs.get('boll_window', 20)
                    std_dev = kwargs.get('boll_std_dev', 2.0)
                    result_df = self._add_bollinger_bands(result_df, window, std_dev)
                elif indicator == "KDJ":
                    k_period = kwargs.get('k_period', 9)
                    d_period = kwargs.get('d_period', 3)
                    j_period = kwargs.get('j_period', 3)
                    result_df = self._add_kdj_indicators(result_df, k_period, d_period, j_period)
                elif indicator == "WILLIAMS":
                    window = kwargs.get('williams_window', 14)
                    result_df = self._add_williams_r(result_df, window)
                elif indicator == "VOLUME_MA":
                    volume_windows = kwargs.get('volume_windows', [5, 10])
                    result_df = self._add_volume_ma(result_df, volume_windows=volume_windows)
                    
            logger.info(f"成功计算{len(indicators)}个技术指标")
            return result_df
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """获取所有可能的特征名称列表"""
        names = []
        
        # MA指标特征名
        for window in [5, 10, 20, 30, 60]:
            names.extend([f'ma_{window}', f'close_ma_{window}_ratio'])
        
        # EMA指标特征名  
        for window in [12, 26]:
            names.extend([f'ema_{window}'])
            
        # RSI指标
        names.extend(['rsi_14', 'rsi_6'])
        
        # MACD指标
        names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # 布林带指标
        names.extend(['boll_upper', 'boll_middle', 'boll_lower', 'boll_width', 'boll_position'])
        
        # KDJ指标
        names.extend(['kdj_k', 'kdj_d', 'kdj_j'])
        
        # 威廉指标
        names.extend(['williams_r'])
        
        # 成交量移动平均
        names.extend(['volume_ma_5', 'volume_ma_10', 'volume_ratio'])
        
        return names
    
    def _prepare_data(self, data: Union[StockData, pd.DataFrame]) -> pd.DataFrame:
        """数据预处理和验证"""
        if isinstance(data, StockData):
            df = data.to_dataframe()
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
        
        if df.empty:
            logger.warning("输入数据为空")
            return df
            
        # 验证必需的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
            
        # 确保数据类型正确
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 移除无效数据
        df = df.dropna(subset=required_cols)
        
        if df.empty:
            logger.warning("数据处理后为空")
            
        return df
    
    def _add_ma_indicators(self, df: pd.DataFrame, 
                          ma_windows: List[int] = None) -> pd.DataFrame:
        """
        添加移动平均线指标
        
        Args:
            df: 数据DataFrame
            ma_windows: MA窗口期列表
            
        Returns:
            pd.DataFrame: 包含MA指标的DataFrame
        """
        if ma_windows is None:
            ma_windows = [5, 10, 20, 30, 60]
            
        for window in ma_windows:
            if window > 0 and window < len(df):
                df[f'ma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
                # 价格相对MA的比例
                df[f'close_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
                
        return df
    
    def _add_ema_indicators(self, df: pd.DataFrame,
                           ema_windows: List[int] = None) -> pd.DataFrame:
        """
        添加指数移动平均线指标
        
        Args:
            df: 数据DataFrame
            ema_windows: EMA窗口期列表
            
        Returns:
            pd.DataFrame: 包含EMA指标的DataFrame
        """
        if ema_windows is None:
            ema_windows = [12, 26]
            
        for window in ema_windows:
            if window > 0:
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
                
        return df
    
    def _add_rsi_indicators(self, df: pd.DataFrame,
                           rsi_windows: List[int] = None) -> pd.DataFrame:
        """
        添加RSI（相对强弱指标）
        
        RSI = 100 - (100 / (1 + RS))
        RS = 平均上涨幅度 / 平均下跌幅度
        
        Args:
            df: 数据DataFrame  
            rsi_windows: RSI窗口期列表
            
        Returns:
            pd.DataFrame: 包含RSI指标的DataFrame
        """
        if rsi_windows is None:
            rsi_windows = [14, 6]
            
        # 计算价格变化
        price_change = df['close'].diff()
        
        for window in rsi_windows:
            if window > 0:
                # 分别计算上涨和下跌
                gains = price_change.where(price_change > 0, 0)
                losses = -price_change.where(price_change < 0, 0)
                
                # 计算平均收益和损失
                avg_gains = gains.rolling(window=window, min_periods=1).mean()
                avg_losses = losses.rolling(window=window, min_periods=1).mean()
                
                # 计算RSI
                rs = avg_gains / avg_losses
                df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
                
        return df
    
    def _add_macd_indicators(self, df: pd.DataFrame,
                            fast_period: int = 12,
                            slow_period: int = 26, 
                            signal_period: int = 9) -> pd.DataFrame:
        """
        添加MACD指标
        
        MACD = EMA(12) - EMA(26)
        Signal = EMA(MACD, 9)
        Histogram = MACD - Signal
        
        Args:
            df: 数据DataFrame
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            pd.DataFrame: 包含MACD指标的DataFrame
        """
        # 计算快慢EMA
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # MACD线
        df['macd'] = ema_fast - ema_slow
        
        # 信号线
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # MACD柱状图
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame,
                            window: int = 20,
                            std_dev: float = 2.0) -> pd.DataFrame:
        """
        添加布林带指标
        
        中轨 = MA(20)
        上轨 = 中轨 + 2 * 标准差
        下轨 = 中轨 - 2 * 标准差
        
        Args:
            df: 数据DataFrame
            window: 窗口期
            std_dev: 标准差倍数
            
        Returns:
            pd.DataFrame: 包含布林带指标的DataFrame
        """
        # 中轨（移动平均）
        df['boll_middle'] = df['close'].rolling(window=window, min_periods=1).mean()
        
        # 标准差
        std = df['close'].rolling(window=window, min_periods=1).std()
        
        # 上轨和下轨
        df['boll_upper'] = df['boll_middle'] + (std_dev * std)
        df['boll_lower'] = df['boll_middle'] - (std_dev * std)
        
        # 带宽
        df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_middle']
        
        # 价格位置（0表示下轨，1表示上轨）
        df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])
        
        return df
    
    def _add_kdj_indicators(self, df: pd.DataFrame,
                           k_period: int = 9,
                           d_period: int = 3,
                           j_period: int = 3) -> pd.DataFrame:
        """
        添加KDJ指标
        
        RSV = (今日收盘价 - N日内最低价) / (N日内最高价 - N日内最低价) * 100
        K = RSV的M1日移动平均
        D = K的M2日移动平均  
        J = 3*K - 2*D
        
        Args:
            df: 数据DataFrame
            k_period: K线周期
            d_period: D线周期
            j_period: J线周期（暂未使用）
            
        Returns:
            pd.DataFrame: 包含KDJ指标的DataFrame
        """
        # 计算RSV
        lowest_low = df['low'].rolling(window=k_period, min_periods=1).min()
        highest_high = df['high'].rolling(window=k_period, min_periods=1).max()
        
        rsv = (df['close'] - lowest_low) / (highest_high - lowest_low) * 100
        
        # K线（RSV的移动平均）
        df['kdj_k'] = rsv.rolling(window=d_period, min_periods=1).mean()
        
        # D线（K线的移动平均）
        df['kdj_d'] = df['kdj_k'].rolling(window=d_period, min_periods=1).mean()
        
        # J线
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        return df
    
    def _add_williams_r(self, df: pd.DataFrame, 
                       window: int = 14) -> pd.DataFrame:
        """
        添加威廉指标（Williams %R）
        
        %R = (最高价 - 收盘价) / (最高价 - 最低价) * (-100)
        
        Args:
            df: 数据DataFrame
            window: 窗口期
            
        Returns:
            pd.DataFrame: 包含威廉指标的DataFrame
        """
        highest_high = df['high'].rolling(window=window, min_periods=1).max()
        lowest_low = df['low'].rolling(window=window, min_periods=1).min()
        
        df['williams_r'] = (highest_high - df['close']) / (highest_high - lowest_low) * (-100)
        
        return df
    
    def _add_volume_ma(self, df: pd.DataFrame,
                      volume_windows: List[int] = None) -> pd.DataFrame:
        """
        添加成交量移动平均指标
        
        Args:
            df: 数据DataFrame
            volume_windows: 成交量MA窗口期列表
            
        Returns:
            pd.DataFrame: 包含成交量指标的DataFrame
        """
        if volume_windows is None:
            volume_windows = [5, 10]
            
        for window in volume_windows:
            if window > 0:
                df[f'volume_ma_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean()
        
        # 当日成交量与5日平均的比率
        if 'volume_ma_5' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume_ma_5']
            
        return df


# 高性能计算函数（使用numba加速）
@jit(nopython=True)
def _fast_rsi_calculation(prices: np.ndarray, window: int) -> np.ndarray:
    """
    使用numba加速RSI计算
    
    Args:
        prices: 价格数组
        window: 窗口期
        
    Returns:
        np.ndarray: RSI值数组
    """
    n = len(prices)
    rsi = np.full(n, np.nan)
    
    if n < window:
        return rsi
    
    # 计算价格变化
    changes = np.diff(prices)
    
    for i in range(window, n):
        # 获取窗口内的价格变化
        window_changes = changes[i-window:i]
        
        # 分别计算上涨和下跌
        gains = np.where(window_changes > 0, window_changes, 0)
        losses = np.where(window_changes < 0, -window_changes, 0)
        
        # 计算平均收益和损失
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # 计算RSI
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_single_indicator(data: Union[StockData, pd.DataFrame], 
                              indicator: str, **kwargs) -> pd.DataFrame:
    """
    计算单个技术指标的便捷函数
    
    Args:
        data: 输入数据
        indicator: 指标名称
        **kwargs: 指标参数
        
    Returns:
        pd.DataFrame: 包含指标的DataFrame
    """
    calculator = TechnicalIndicators()
    return calculator.calculate(data, indicators=[indicator], **kwargs)
