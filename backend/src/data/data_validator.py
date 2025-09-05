"""
数据验证器 - 简化版本

提供数据质量检查和验证功能，不依赖复杂的数据模型。
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """数据质量验证器 - 简化版本"""
    
    def __init__(self):
        """初始化验证器"""
        self.required_columns = ['date', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']
        self.numeric_columns = ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 'amount']
    
    def validate_stock_dataframe(self, df: pd.DataFrame, symbol: str = None) -> bool:
        """
        验证股票数据DataFrame的质量
        
        Args:
            df: 股票数据DataFrame
            symbol: 股票代码（可选）
            
        Returns:
            bool: 验证是否通过
        """
        if df is None or df.empty:
            logger.warning(f"数据为空 - {symbol}")
            return False
        
        try:
            # 检查必需列
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"缺少必需列 - {symbol}: {missing_columns}")
                return False
            
            # 检查数据类型
            for col in self.numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        logger.warning(f"列 {col} 不是数值类型 - {symbol}")
                        return False
            
            # 检查缺失值比例
            total_rows = len(df)
            for col in self.required_columns:
                missing_count = df[col].isna().sum()
                missing_ratio = missing_count / total_rows
                if missing_ratio > 0.1:  # 超过10%缺失值
                    logger.warning(f"列 {col} 缺失值过多 ({missing_ratio:.1%}) - {symbol}")
                    return False
            
            # 检查价格逻辑
            if not self._validate_price_logic(df):
                logger.warning(f"价格逻辑验证失败 - {symbol}")
                return False
            
            # 检查日期连续性
            if not self._validate_date_continuity(df):
                logger.warning(f"日期连续性验证失败 - {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证时出错 - {symbol}: {e}")
            return False
    
    def _validate_price_logic(self, df: pd.DataFrame) -> bool:
        """验证价格逻辑"""
        try:
            # 检查价格范围
            price_columns = ['open_price', 'close_price', 'high_price', 'low_price']
            for col in price_columns:
                if col in df.columns:
                    prices = df[col].dropna()
                    if (prices <= 0).any():
                        return False
                    if (prices > 10000).any():  # 价格不应超过10000元
                        return False
            
            # 检查高低价逻辑
            if all(col in df.columns for col in ['high_price', 'low_price', 'open_price', 'close_price']):
                # 最高价 >= 最低价
                invalid_high_low = (df['high_price'] < df['low_price']).any()
                if invalid_high_low:
                    return False
                
                # 开盘价和收盘价应在高低价范围内
                open_out_range = ((df['open_price'] > df['high_price']) | 
                                (df['open_price'] < df['low_price'])).any()
                close_out_range = ((df['close_price'] > df['high_price']) | 
                                 (df['close_price'] < df['low_price'])).any()
                
                if open_out_range or close_out_range:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"价格逻辑验证出错: {e}")
            return False
    
    def _validate_date_continuity(self, df: pd.DataFrame) -> bool:
        """验证日期连续性"""
        try:
            if 'date' not in df.columns or len(df) <= 1:
                return True
            
            # 确保日期列是日期类型
            df_sorted = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_sorted['date']):
                df_sorted['date'] = pd.to_datetime(df_sorted['date'])
            
            # 按日期排序
            df_sorted = df_sorted.sort_values('date')
            
            # 检查是否有重复日期
            if df_sorted['date'].duplicated().any():
                return False
            
            # 检查日期是否在合理范围内
            min_date = df_sorted['date'].min()
            max_date = df_sorted['date'].max()
            
            if min_date < pd.Timestamp('1990-01-01'):
                return False
            if max_date > pd.Timestamp.now() + pd.Timedelta(days=1):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"日期连续性验证出错: {e}")
            return False
    
    def get_data_quality_summary(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        获取数据质量摘要
        
        Args:
            df: 股票数据DataFrame
            symbol: 股票代码
            
        Returns:
            Dict: 数据质量摘要
        """
        if df is None or df.empty:
            return {
                "symbol": symbol,
                "is_valid": False,
                "total_records": 0,
                "issues": ["数据为空"]
            }
        
        try:
            issues = []
            total_records = len(df)
            
            # 检查列完整性
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"缺少列: {', '.join(missing_columns)}")
            
            # 检查缺失值
            for col in self.required_columns:
                if col in df.columns:
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        missing_ratio = missing_count / total_records
                        issues.append(f"{col}列缺失{missing_count}个值 ({missing_ratio:.1%})")
            
            # 检查数据范围
            if 'date' in df.columns:
                date_range = {
                    "start": str(df['date'].min()),
                    "end": str(df['date'].max())
                }
            else:
                date_range = None
            
            # 计算基本统计
            price_stats = {}
            if 'close_price' in df.columns:
                close_prices = df['close_price'].dropna()
                if not close_prices.empty:
                    price_stats = {
                        "mean": float(close_prices.mean()),
                        "min": float(close_prices.min()),
                        "max": float(close_prices.max()),
                        "std": float(close_prices.std())
                    }
            
            return {
                "symbol": symbol,
                "is_valid": len(issues) == 0,
                "total_records": total_records,
                "date_range": date_range,
                "price_stats": price_stats,
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"生成数据质量摘要出错 - {symbol}: {e}")
            return {
                "symbol": symbol,
                "is_valid": False,
                "total_records": 0,
                "issues": [f"处理错误: {str(e)}"]
            }
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理DataFrame数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            DataFrame: 清理后的DataFrame
        """
        if df is None or df.empty:
            return df
        
        try:
            cleaned_df = df.copy()
            
            # 删除完全空的行
            cleaned_df = cleaned_df.dropna(how='all')
            
            # 处理数值列的异常值
            for col in self.numeric_columns:
                if col in cleaned_df.columns:
                    # 替换无穷大和负无穷大值
                    cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # 对于价格列，删除负值
                    if 'price' in col:
                        cleaned_df.loc[cleaned_df[col] <= 0, col] = np.nan
            
            # 确保日期列的格式
            if 'date' in cleaned_df.columns:
                cleaned_df['date'] = pd.to_datetime(cleaned_df['date']).dt.date
            
            # 按日期排序
            if 'date' in cleaned_df.columns:
                cleaned_df = cleaned_df.sort_values('date')
                # 删除重复日期，保留最后一个
                cleaned_df = cleaned_df.drop_duplicates(subset=['date'], keep='last')
            
            # 重置索引
            cleaned_df = cleaned_df.reset_index(drop=True)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"清理DataFrame出错: {e}")
            return df
    
    def is_data_sufficient(self, df: pd.DataFrame, min_records: int = 5) -> bool:
        """
        检查数据是否足够用于分析
        
        Args:
            df: DataFrame
            min_records: 最小记录数
            
        Returns:
            bool: 数据是否足够
        """
        if df is None or df.empty:
            return False
        
        return len(df) >= min_records
    
    def validate_date_range(self, start_date: date, end_date: date) -> bool:
        """
        验证日期范围是否合理
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 日期范围是否合理
        """
        try:
            # 检查日期顺序
            if start_date > end_date:
                return False
            
            # 检查日期范围
            min_allowed_date = date(1990, 1, 1)
            max_allowed_date = date.today()
            
            if start_date < min_allowed_date or start_date > max_allowed_date:
                return False
            
            if end_date < min_allowed_date or end_date > max_allowed_date:
                return False
            
            # 检查日期跨度不能太大（最多5年）
            if (end_date - start_date).days > 365 * 5:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证日期范围出错: {e}")
            return False
