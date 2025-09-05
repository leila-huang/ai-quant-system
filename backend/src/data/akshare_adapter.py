"""
AKShare数据适配器 - 简化版本

从AKShare获取A股数据的简化实现，减少复杂依赖。
"""

import time
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
import logging

import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor

from ..models.basic_models import StockData, StockDailyBar
from .data_validator import DataValidator

logger = logging.getLogger(__name__)


class AKShareAdapter:
    """AKShare数据适配器 - 简化版本"""
    
    def __init__(self, 
                 executor_pool_size: int = 4, 
                 rate_limit_seconds: float = 0.2,
                 max_retries: int = 3):
        """
        初始化适配器
        """
        self.executor_pool_size = executor_pool_size
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.validator = DataValidator()
        
        # 线程池，用于并发API调用
        self.executor = ThreadPoolExecutor(max_workers=executor_pool_size)
        
        # 记录请求时间，用于限速
        self.last_request_time = 0
    
    def _rate_limit(self):
        """实现请求限速"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _retry_request(self, func, *args, **kwargs):
        """带重试的请求执行"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"请求失败，{wait_time:.1f}秒后重试 (第{attempt+1}次): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"请求最终失败，已重试{self.max_retries}次: {e}")
        
        raise last_exception
    
    def get_stock_daily_data(self, 
                           symbol: str, 
                           start_date: Optional[date] = None, 
                           end_date: Optional[date] = None,
                           adjust: str = "qfq") -> pd.DataFrame:
        """
        获取股票日线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型，qfq-前复权，hfq-后复权，None-不复权
            
        Returns:
            DataFrame: 股票日线数据
        """
        # 设置默认日期
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        # 格式化日期
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        logger.info(f"获取股票 {symbol} 数据，日期范围: {start_str} - {end_str}")
        
        def _fetch_data():
            return ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=adjust
            )
        
        try:
            df = self._retry_request(_fetch_data)
            
            if df is None or df.empty:
                logger.warning(f"股票 {symbol} 未获取到数据")
                return pd.DataFrame()
            
            # 标准化列名
            df = self._standardize_columns(df)
            
            # 数据验证
            if not self.validator.validate_stock_dataframe(df):
                logger.warning(f"股票 {symbol} 数据验证失败")
                return pd.DataFrame()
            
            logger.info(f"成功获取股票 {symbol} 数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 数据失败: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame列名"""
        if df.empty:
            return df
        
        # 标准列名映射
        column_mapping = {
            '日期': 'date',
            '开盘': 'open_price',
            '收盘': 'close_price', 
            '最高': 'high_price',
            '最低': 'low_price',
            '成交量': 'volume',
            '成交额': 'amount',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 确保日期列是datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        
        # 确保数值列是float类型
        numeric_columns = ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 'amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 只保留需要的列
        required_columns = ['date', 'open_price', 'close_price', 'high_price', 'low_price', 'volume', 'amount']
        available_columns = [col for col in required_columns if col in df.columns]
        df = df[available_columns]
        
        return df
    
    def get_stock_data(self, 
                      symbol: str, 
                      start_date: Optional[date] = None, 
                      end_date: Optional[date] = None) -> Optional[StockData]:
        """
        获取股票数据并转换为StockData模型
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            StockData: 股票数据对象
        """
        try:
            df = self.get_stock_daily_data(symbol, start_date, end_date)
            
            if df.empty:
                return None
            
            # 转换为StockDailyBar列表
            bars = []
            for _, row in df.iterrows():
                try:
                    bar = StockDailyBar(
                        date=row['date'],
                        open_price=float(row['open_price']) if pd.notna(row['open_price']) else 0.0,
                        close_price=float(row['close_price']) if pd.notna(row['close_price']) else 0.0,
                        high_price=float(row['high_price']) if pd.notna(row['high_price']) else 0.0,
                        low_price=float(row['low_price']) if pd.notna(row['low_price']) else 0.0,
                        volume=float(row['volume']) if pd.notna(row['volume']) else 0.0,
                        amount=float(row['amount']) if pd.notna(row['amount']) else None
                    )
                    bars.append(bar)
                except Exception as e:
                    logger.warning(f"转换数据行失败: {e}, 跳过该行")
                    continue
            
            if not bars:
                return None
            
            return StockData(
                symbol=symbol,
                name=f"股票{symbol}",  # 简化的名称
                bars=bars
            )
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 数据失败: {e}")
            return None
    
    def get_stock_list(self, market: str = "all") -> List[Dict]:
        """
        获取股票列表
        
        Args:
            market: 市场类型，all-全部，sh-上海，sz-深圳
            
        Returns:
            List[Dict]: 股票基本信息列表
        """
        try:
            def _fetch_stock_list():
                return ak.stock_info_a_code_name()
            
            df = self._retry_request(_fetch_stock_list)
            
            if df is None or df.empty:
                logger.warning("获取股票列表失败")
                return []
            
            # 转换为字典列表
            stock_list = []
            for _, row in df.iterrows():
                try:
                    code = str(row['code']).zfill(6)  # 确保代码为6位数字
                    name = str(row['name'])
                    
                    # 根据市场过滤
                    if market == "sh" and not code.startswith(('60', '68', '90')):
                        continue
                    elif market == "sz" and not code.startswith(('00', '30', '20')):
                        continue
                    
                    stock_info = {
                        'symbol': code,
                        'name': name,
                        'exchange': 'SH' if code.startswith(('60', '68', '90')) else 'SZ'
                    }
                    stock_list.append(stock_info)
                    
                except Exception as e:
                    logger.warning(f"处理股票信息失败: {e}")
                    continue
            
            logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")
            return stock_list
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
