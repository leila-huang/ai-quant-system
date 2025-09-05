"""
Parquet存储引擎

提供高性能的列式数据存储，支持分片、压缩、增量更新和快速查询。
"""

import os
import threading
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from loguru import logger

from backend.src.models.basic_models import StockData, StockDailyBar


class ParquetStorage:
    """
    Parquet存储引擎
    
    特性:
    - 按年份和股票代码分片存储
    - 支持增量更新和追加写入
    - Snappy压缩算法
    - 并发读写支持
    - 数据完整性校验
    """
    
    def __init__(self, base_path: str = "data/parquet", 
                 compression: str = "snappy",
                 max_workers: int = 4):
        """
        初始化Parquet存储引擎
        
        Args:
            base_path: 数据存储根目录
            compression: 压缩算法 (snappy, gzip, brotli)
            max_workers: 最大并发工作线程数
        """
        self.base_path = Path(base_path)
        self.compression = compression
        self.max_workers = max_workers
        
        # 确保基础目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 文件锁管理
        self._file_locks: Dict[str, threading.RLock] = {}
        self._lock_manager_lock = threading.Lock()
        
        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"ParquetStorage initialized: base_path={base_path}, "
                   f"compression={compression}, max_workers={max_workers}")
    
    def _get_file_lock(self, file_path: str) -> threading.RLock:
        """获取文件锁，确保并发安全"""
        with self._lock_manager_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.RLock()
            return self._file_locks[file_path]
    
    def _get_file_path(self, symbol: str, year: int) -> Path:
        """
        获取股票数据文件路径
        格式: data/parquet/{year}/{symbol}/daily.parquet
        """
        return self.base_path / str(year) / symbol / "daily.parquet"
    
    def _ensure_directory(self, file_path: Path):
        """确保目录存在"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """计算数据哈希值，用于完整性校验"""
        data_str = df.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _stock_data_to_dataframe(self, stock_data: StockData) -> pd.DataFrame:
        """将StockData转换为DataFrame"""
        data = []
        for bar in stock_data.bars:
            row = {
                'date': bar.date,
                'open': bar.open_price,
                'close': bar.close_price,
                'high': bar.high_price,
                'low': bar.low_price,
                'volume': bar.volume,
                'amount': bar.amount,
                'adjust_factor': bar.adjust_factor
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df['symbol'] = stock_data.symbol
        df['name'] = stock_data.name or ''
        
        # 确保日期列为datetime类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _dataframe_to_stock_data(self, df: pd.DataFrame, symbol: str) -> StockData:
        """将DataFrame转换为StockData"""
        bars = []
        for _, row in df.iterrows():
            bar = StockDailyBar(
                date=row['date'].date() if hasattr(row['date'], 'date') else row['date'],
                open_price=float(row['open']),
                close_price=float(row['close']),
                high_price=float(row['high']),
                low_price=float(row['low']),
                volume=float(row['volume']),
                amount=float(row['amount']) if pd.notna(row['amount']) else None,
                adjust_factor=float(row['adjust_factor']) if pd.notna(row['adjust_factor']) else None
            )
            bars.append(bar)
        
        return StockData(
            symbol=symbol,
            name=df['name'].iloc[0] if 'name' in df.columns and len(df) > 0 else None,
            bars=bars
        )
    
    def save_stock_data(self, stock_data: StockData, update_mode: str = "append") -> bool:
        """
        保存股票数据到Parquet文件
        
        Args:
            stock_data: 股票数据
            update_mode: 更新模式 ('append', 'overwrite', 'merge')
                - append: 追加新数据
                - overwrite: 覆盖整个文件
                - merge: 合并数据，去重并排序
        
        Returns:
            保存是否成功
        """
        try:
            if not stock_data.bars:
                logger.warning(f"No data to save for symbol {stock_data.symbol}")
                return False
            
            # 转换为DataFrame
            new_df = self._stock_data_to_dataframe(stock_data)
            
            # 按年份分组保存
            year_groups = new_df.groupby(new_df['date'].dt.year)
            
            success_count = 0
            total_years = len(year_groups)
            
            for year, year_df in year_groups:
                file_path = self._get_file_path(stock_data.symbol, year)
                lock = self._get_file_lock(str(file_path))
                
                with lock:
                    try:
                        self._ensure_directory(file_path)
                        
                        # 如果文件存在且不是覆盖模式，需要合并数据
                        if file_path.exists() and update_mode != "overwrite":
                            existing_df = pd.read_parquet(file_path)
                            
                            if update_mode == "append":
                                # 简单追加
                                combined_df = pd.concat([existing_df, year_df], ignore_index=True)
                            elif update_mode == "merge":
                                # 合并并去重
                                combined_df = pd.concat([existing_df, year_df], ignore_index=True)
                                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                                combined_df = combined_df.sort_values('date').reset_index(drop=True)
                        else:
                            combined_df = year_df
                        
                        # 保存到Parquet
                        combined_df.to_parquet(
                            file_path,
                            compression=self.compression,
                            index=False,
                            engine='pyarrow'
                        )
                        
                        # 验证数据完整性
                        if self._verify_file_integrity(file_path):
                            success_count += 1
                            logger.info(f"Successfully saved {len(combined_df)} records "
                                      f"for {stock_data.symbol} year {year}")
                        else:
                            logger.error(f"Data integrity check failed for {file_path}")
                            return False
                            
                    except Exception as e:
                        logger.error(f"Error saving data for {stock_data.symbol} year {year}: {e}")
                        return False
            
            return success_count == total_years
            
        except Exception as e:
            logger.error(f"Error saving stock data for {stock_data.symbol}: {e}")
            return False
    
    def load_stock_data(self, symbol: str, 
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> Optional[StockData]:
        """
        加载股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            股票数据，如果没有数据则返回None
        """
        try:
            # 确定需要加载的年份范围
            if start_date and end_date:
                start_year = start_date.year
                end_year = end_date.year
            else:
                # 扫描所有可能的年份
                symbol_dir = self.base_path.glob(f"*/{symbol}")
                years = []
                for path in symbol_dir:
                    try:
                        year = int(path.parent.name)
                        years.append(year)
                    except ValueError:
                        continue
                
                if not years:
                    logger.info(f"No data found for symbol {symbol}")
                    return None
                
                start_year = min(years)
                end_year = max(years)
            
            # 并行加载各年份数据
            dataframes = []
            year_range = range(start_year, end_year + 1)
            
            def load_year_data(year):
                file_path = self._get_file_path(symbol, year)
                if not file_path.exists():
                    return None
                
                lock = self._get_file_lock(str(file_path))
                with lock:
                    try:
                        df = pd.read_parquet(file_path)
                        return df
                    except Exception as e:
                        logger.error(f"Error loading data for {symbol} year {year}: {e}")
                        return None
            
            # 使用线程池并行加载
            future_to_year = {}
            for year in year_range:
                future = self._executor.submit(load_year_data, year)
                future_to_year[future] = year
            
            for future in as_completed(future_to_year):
                df = future.result()
                if df is not None:
                    dataframes.append(df)
            
            if not dataframes:
                logger.info(f"No data loaded for symbol {symbol}")
                return None
            
            # 合并所有年份的数据
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            
            # 应用日期过滤
            if start_date:
                combined_df = combined_df[combined_df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                combined_df = combined_df[combined_df['date'] <= pd.to_datetime(end_date)]
            
            if combined_df.empty:
                logger.info(f"No data in date range for symbol {symbol}")
                return None
            
            # 转换为StockData对象
            stock_data = self._dataframe_to_stock_data(combined_df, symbol)
            
            logger.info(f"Loaded {len(stock_data.bars)} bars for {symbol}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error loading stock data for {symbol}: {e}")
            return None
    
    def _verify_file_integrity(self, file_path: Path) -> bool:
        """验证文件完整性"""
        try:
            # 尝试读取文件
            df = pd.read_parquet(file_path)
            
            # 检查必要的列是否存在
            required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns in {file_path}: {missing_columns}")
                return False
            
            # 检查数据基本合理性
            if df.empty:
                logger.error(f"Empty data in {file_path}")
                return False
            
            # 检查价格数据合理性
            price_columns = ['open', 'close', 'high', 'low']
            for col in price_columns:
                if (df[col] <= 0).any():
                    logger.error(f"Invalid prices found in {file_path} column {col}")
                    return False
            
            # 检查高低价关系
            if (df['high'] < df['low']).any():
                logger.error(f"High price less than low price in {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying file integrity for {file_path}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            stats = {
                'total_files': 0,
                'total_size_mb': 0,
                'symbols_count': 0,
                'year_range': None,
                'compression_ratio': 0
            }
            
            symbols = set()
            years = set()
            total_size = 0
            
            # 遍历所有Parquet文件
            for parquet_file in self.base_path.rglob("*.parquet"):
                stats['total_files'] += 1
                total_size += parquet_file.stat().st_size
                
                # 提取股票代码和年份
                parts = parquet_file.parts
                if len(parts) >= 3:
                    year_str = parts[-3]  # 年份目录
                    symbol = parts[-2]    # 股票代码目录
                    
                    try:
                        year = int(year_str)
                        years.add(year)
                        symbols.add(symbol)
                    except ValueError:
                        continue
            
            stats['total_size_mb'] = total_size / (1024 * 1024)
            stats['symbols_count'] = len(symbols)
            stats['year_range'] = (min(years), max(years)) if years else None
            
            # 估算压缩比 (假设未压缩数据大约是压缩后的2-3倍)
            stats['compression_ratio'] = 0.6  # 经验值，实际需要更精确计算
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def delete_symbol_data(self, symbol: str, 
                          year: Optional[int] = None) -> bool:
        """
        删除指定股票的数据
        
        Args:
            symbol: 股票代码
            year: 指定年份，如果为None则删除所有年份
        
        Returns:
            删除是否成功
        """
        try:
            if year:
                # 删除特定年份
                file_path = self._get_file_path(symbol, year)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted data for {symbol} year {year}")
                    return True
            else:
                # 删除所有年份
                deleted_count = 0
                for year_dir in self.base_path.iterdir():
                    if year_dir.is_dir():
                        symbol_dir = year_dir / symbol
                        if symbol_dir.exists():
                            for file in symbol_dir.iterdir():
                                file.unlink()
                            symbol_dir.rmdir()
                            deleted_count += 1
                
                logger.info(f"Deleted {deleted_count} year files for {symbol}")
                return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting data for {symbol}: {e}")
            return False
    
    def list_available_symbols(self) -> List[str]:
        """列出所有可用的股票代码"""
        symbols = set()
        try:
            for year_dir in self.base_path.iterdir():
                if year_dir.is_dir():
                    for symbol_dir in year_dir.iterdir():
                        if symbol_dir.is_dir() and (symbol_dir / "daily.parquet").exists():
                            symbols.add(symbol_dir.name)
            
            return sorted(list(symbols))
        except Exception as e:
            logger.error(f"Error listing available symbols: {e}")
            return []
    
    def close(self):
        """关闭存储引擎，清理资源"""
        try:
            self._executor.shutdown(wait=True)
            logger.info("ParquetStorage closed successfully")
        except Exception as e:
            logger.error(f"Error closing ParquetStorage: {e}")


# 全局实例（单例模式）
_global_storage_instance: Optional[ParquetStorage] = None


def get_parquet_storage(base_path: str = "data/parquet") -> ParquetStorage:
    """获取全局Parquet存储实例"""
    global _global_storage_instance
    
    if _global_storage_instance is None:
        _global_storage_instance = ParquetStorage(base_path=base_path)
    
    return _global_storage_instance
