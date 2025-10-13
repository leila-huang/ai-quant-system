"""
AKShare数据适配器 - 简化版本

从AKShare获取A股数据的简化实现，减少复杂依赖。
"""

import time
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Union
import logging

import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor

from ..models.basic_models import StockData, StockDailyBar
from .data_validator import DataValidator


SAMPLE_STOCK_LIST = [
    {"symbol": "000001", "name": "平安银行", "exchange": "SZ"},
    {"symbol": "000002", "name": "万 科Ａ", "exchange": "SZ"},
    {"symbol": "600000", "name": "浦发银行", "exchange": "SH"},
    {"symbol": "600519", "name": "贵州茅台", "exchange": "SH"},
]

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

    def _normalize_date(self, value: Optional[Union[date, datetime, str]]) -> Optional[date]:
        """将输入转换为 ``date`` 类型，无法解析时返回 ``None``。"""

        if value is None:
            return None

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                logger.warning(f"无法解析日期字符串: {value}")
                return None

        logger.warning(f"不支持的日期类型: {type(value)}")
        return None
    
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
    
    def get_stock_daily_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        adjust: str = "qfq",
    ) -> Optional[StockData]:
        """获取股票日线数据并转换为领域模型。"""

        df = self._fetch_stock_daily_dataframe(symbol, start_date, end_date, adjust)

        if df is None or df.empty:
            logger.warning(f"未能获取 {symbol} 的日线数据")
            return StockData(symbol=symbol, name=f"股票{symbol}", bars=[])

        bars: List[StockDailyBar] = []
        for _, row in df.iterrows():
            try:
                bars.append(
                    StockDailyBar(
                        date=row["date"],
                        open_price=float(row["open_price"]) if pd.notna(row["open_price"]) else 0.0,
                        close_price=float(row["close_price"]) if pd.notna(row["close_price"]) else 0.0,
                        high_price=float(row["high_price"]) if pd.notna(row["high_price"]) else 0.0,
                        low_price=float(row["low_price"]) if pd.notna(row["low_price"]) else 0.0,
                        volume=float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else 0.0,
                        amount=float(row["amount"]) if "amount" in row and pd.notna(row["amount"]) else None,
                    )
                )
            except Exception as exc:
                logger.warning(f"转换 {symbol} 的数据行失败: {exc}")

        if not bars:
            logger.warning(f"{symbol} 的日线数据为空")
            return StockData(symbol=symbol, name=f"股票{symbol}", bars=[])

        return StockData(symbol=symbol, name=f"股票{symbol}", bars=bars)

    def _fetch_stock_daily_dataframe(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取股票日线数据的原始DataFrame。"""
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)

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
                logger.warning(f"股票 {symbol} 未获取到数据，使用示例数据降级")
                return self._generate_sample_dataframe(symbol, start_date, end_date)

            # 标准化列名
            df = self._standardize_columns(df)

            # 数据验证
            if not self.validator.validate_stock_dataframe(df):
                logger.warning(f"股票 {symbol} 数据验证失败，使用示例数据降级")
                return self._generate_sample_dataframe(symbol, start_date, end_date)

            logger.info(f"成功获取股票 {symbol} 数据，共 {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"获取股票 {symbol} 数据失败: {e}，使用示例数据降级")
            return self._generate_sample_dataframe(symbol, start_date, end_date)
    
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

    def _generate_sample_dataframe(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """生成离线降级使用的示例数据。"""

        rows = []
        rng = random.Random(symbol)
        current = start_date
        price = rng.uniform(20, 120)

        while current <= end_date:
            change = rng.uniform(-0.03, 0.03)
            open_price = round(price * (1 + change), 2)
            close_price = round(open_price * (1 + rng.uniform(-0.02, 0.02)), 2)
            high_price = round(max(open_price, close_price) * (1 + rng.uniform(0, 0.01)), 2)
            low_price = round(min(open_price, close_price) * (1 - rng.uniform(0, 0.01)), 2)
            volume = int(rng.uniform(5e5, 2e6))
            amount = float(volume * ((open_price + close_price) / 2))

            rows.append(
                {
                    "date": current,
                    "open_price": open_price,
                    "close_price": close_price,
                    "high_price": high_price,
                    "low_price": low_price,
                    "volume": volume,
                    "amount": amount,
                }
            )

            current += timedelta(days=1)
            price = close_price

        return pd.DataFrame(rows)
    
    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        adjust: str = "qfq",
    ) -> Optional[StockData]:
        """向后兼容的股票数据获取接口。"""

        try:
            return self.get_stock_daily_data(symbol, start_date, end_date, adjust)
        except Exception as exc:
            logger.error(f"获取股票 {symbol} 数据失败: {exc}")
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
                logger.warning("获取股票列表失败，使用示例列表降级")
                return [stock.copy() for stock in SAMPLE_STOCK_LIST]
            
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
            
            if not stock_list:
                logger.warning("股票列表为空，使用示例列表降级")
                return [stock.copy() for stock in SAMPLE_STOCK_LIST]

            logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")
            return stock_list

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}，使用示例列表降级")
            return [stock.copy() for stock in SAMPLE_STOCK_LIST]
    
    def get_stock_fundamental_data(self, symbol: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        获取股票财务数据
        
        Args:
            symbol: 股票代码
            year: 财报年份（默认最近一年）
            
        Returns:
            Dict: 财务指标数据
        """
        try:
            if year is None:
                year = datetime.now().year
            
            def _fetch_fundamental():
                # 获取财务指标
                return ak.stock_financial_analysis_indicator(symbol=symbol)
            
            df = self._retry_request(_fetch_fundamental)
            
            if df is None or df.empty:
                logger.warning(f"股票 {symbol} 未获取到财务数据")
                return None
            
            # 筛选最近一年的数据
            if '季度' in df.columns:
                df = df.sort_values('季度', ascending=False)
                latest = df.iloc[0]
                
                fundamental_data = {
                    'symbol': symbol,
                    'date': latest.get('季度'),
                    'eps': float(latest.get('每股收益', 0)),  # 每股收益
                    'roe': float(latest.get('净资产收益率', 0)),  # 净资产收益率
                    'operating_revenue': float(latest.get('营业收入', 0)),  # 营业收入
                    'net_profit': float(latest.get('净利润', 0)),  # 净利润
                    'total_assets': float(latest.get('总资产', 0)),  # 总资产
                    'total_liabilities': float(latest.get('总负债', 0)),  # 总负债
                    'gross_profit_margin': float(latest.get('销售毛利率', 0)),  # 毛利率
                    'debt_to_asset_ratio': float(latest.get('资产负债比率', 0)),  # 资产负债率
                }
                
                logger.info(f"成功获取股票 {symbol} 财务数据")
                return fundamental_data
            
            return None
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 财务数据失败: {e}")
            return None
    
    def get_stock_industry_classification(self, symbol: str) -> Optional[Dict]:
        """
        获取股票行业分类
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 行业分类信息
        """
        try:
            def _fetch_industry():
                # 获取行业分类
                return ak.stock_individual_info_em(symbol=symbol)
            
            df = self._retry_request(_fetch_industry)
            
            if df is None or df.empty:
                logger.warning(f"股票 {symbol} 未获取到行业分类")
                return None
            
            # 提取行业信息
            industry_data = {}
            for _, row in df.iterrows():
                item = row.get('item', '')
                value = row.get('value', '')
                
                if '行业' in item:
                    industry_data['industry'] = str(value)
                elif '板块' in item or '概念' in item:
                    if 'sectors' not in industry_data:
                        industry_data['sectors'] = []
                    industry_data['sectors'].append(str(value))
            
            if industry_data:
                industry_data['symbol'] = symbol
                logger.info(f"成功获取股票 {symbol} 行业分类: {industry_data.get('industry', 'Unknown')}")
                return industry_data
            
            return None
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 行业分类失败: {e}")
            return None
    
    def get_stock_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """
        获取股票实时行情
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 实时行情数据
        """
        try:
            def _fetch_realtime():
                return ak.stock_zh_a_spot_em()
            
            df = self._retry_request(_fetch_realtime)
            
            if df is None or df.empty:
                logger.warning("获取实时行情失败")
                return None
            
            # 筛选指定股票
            stock_df = df[df['代码'] == symbol]
            
            if stock_df.empty:
                logger.warning(f"未找到股票 {symbol} 的实时行情")
                return None
            
            row = stock_df.iloc[0]
            
            quote_data = {
                'symbol': symbol,
                'name': str(row.get('名称', '')),
                'latest_price': float(row.get('最新价', 0)),
                'change_pct': float(row.get('涨跌幅', 0)),
                'change_amount': float(row.get('涨跌额', 0)),
                'volume': float(row.get('成交量', 0)),
                'amount': float(row.get('成交额', 0)),
                'open_price': float(row.get('今开', 0)),
                'high_price': float(row.get('最高', 0)),
                'low_price': float(row.get('最低', 0)),
                'pre_close': float(row.get('昨收', 0)),
                'timestamp': datetime.now()
            }
            
            logger.debug(f"获取股票 {symbol} 实时行情: {quote_data['latest_price']}")
            return quote_data
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 实时行情失败: {e}")
            return None
    
    def get_trading_calendar(self, year: Optional[int] = None) -> List[date]:
        """
        获取交易日历
        
        Args:
            year: 年份（默认当年）
            
        Returns:
            List[date]: 交易日列表
        """
        try:
            if year is None:
                year = datetime.now().year
            
            def _fetch_calendar():
                return ak.tool_trade_date_hist_sina()
            
            df = self._retry_request(_fetch_calendar)
            
            if df is None or df.empty:
                logger.warning("获取交易日历失败")
                return []
            
            # 筛选指定年份
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            year_df = df[df['trade_date'].dt.year == year]
            
            trading_days = [d.date() for d in year_df['trade_date']]
            trading_days.sort()
            
            logger.info(f"{year}年交易日历: 共 {len(trading_days)} 个交易日")
            return trading_days
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []
    
    def get_index_data(self, 
                      index_code: str = "000001",  # 上证指数
                      start_date: Optional[date] = None,
                      end_date: Optional[date] = None) -> pd.DataFrame:
        """
        获取指数数据
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 指数数据
        """
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=365)
            if not end_date:
                end_date = date.today()
            
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            def _fetch_index():
                return ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            
            df = self._retry_request(_fetch_index)
            
            if df is None or df.empty:
                logger.warning(f"指数 {index_code} 未获取到数据")
                return pd.DataFrame()
            
            # 筛选日期范围
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= pd.to_datetime(start_date)) & 
                   (df['date'] <= pd.to_datetime(end_date))]
            
            # 标准化列名
            df = df.rename(columns={
                'close': 'close_price',
                'open': 'open_price',
                'high': 'high_price',
                'low': 'low_price'
            })
            
            logger.info(f"成功获取指数 {index_code} 数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取指数 {index_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
