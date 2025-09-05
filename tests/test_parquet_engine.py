"""
Parquet存储引擎测试

测试ParquetStorage类的各项功能：
- 数据保存和加载
- 分片存储
- 增量更新
- 数据压缩
- 并发读写
- 数据完整性校验
"""

import pytest
import pandas as pd
import tempfile
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List
import threading
import time
import concurrent.futures

from backend.src.storage.parquet_engine import ParquetStorage, get_parquet_storage
from backend.src.models.simple_models import StockData, StockDailyBar


class TestParquetStorage:
    """Parquet存储引擎测试类"""
    
    @pytest.fixture
    def temp_storage_path(self):
        """创建临时存储目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def parquet_storage(self, temp_storage_path):
        """创建ParquetStorage实例"""
        storage = ParquetStorage(base_path=temp_storage_path)
        yield storage
        storage.close()
    
    @pytest.fixture
    def sample_stock_data(self):
        """创建示例股票数据"""
        start_date = date(2023, 1, 1)
        bars = []
        
        for i in range(100):  # 100天的数据
            current_date = start_date + timedelta(days=i)
            # 跳过周末
            if current_date.weekday() >= 5:
                continue
                
            bar = StockDailyBar(
                date=current_date,
                open=100.0 + i * 0.1,
                close=100.5 + i * 0.1,
                high=101.0 + i * 0.1,
                low=99.5 + i * 0.1,
                volume=1000000 + i * 1000,
                amount=100000000 + i * 100000,
                adjust_factor=1.0
            )
            bars.append(bar)
        
        return StockData(
            symbol="000001",
            name="平安银行",
            bars=bars
        )
    
    @pytest.fixture
    def multi_year_stock_data(self):
        """创建跨年股票数据"""
        bars = []
        
        # 2022年数据
        for i in range(50):
            current_date = date(2022, 1, 1) + timedelta(days=i * 2)
            if current_date.year > 2022:
                break
            bar = StockDailyBar(
                date=current_date,
                open=90.0 + i * 0.1,
                close=90.5 + i * 0.1,
                high=91.0 + i * 0.1,
                low=89.5 + i * 0.1,
                volume=900000 + i * 1000,
                amount=90000000 + i * 100000,
                adjust_factor=1.0
            )
            bars.append(bar)
        
        # 2023年数据
        for i in range(50):
            current_date = date(2023, 1, 1) + timedelta(days=i * 2)
            if current_date.year > 2023:
                break
            bar = StockDailyBar(
                date=current_date,
                open=100.0 + i * 0.1,
                close=100.5 + i * 0.1,
                high=101.0 + i * 0.1,
                low=99.5 + i * 0.1,
                volume=1000000 + i * 1000,
                amount=100000000 + i * 100000,
                adjust_factor=1.0
            )
            bars.append(bar)
        
        return StockData(
            symbol="000002",
            name="万科A",
            bars=bars
        )
    
    def test_basic_save_and_load(self, parquet_storage, sample_stock_data):
        """测试基本的保存和加载功能"""
        # 保存数据
        result = parquet_storage.save_stock_data(sample_stock_data, update_mode="overwrite")
        assert result is True
        
        # 加载数据
        loaded_data = parquet_storage.load_stock_data("000001")
        assert loaded_data is not None
        assert loaded_data.symbol == "000001"
        assert loaded_data.name == "平安银行"
        assert len(loaded_data.bars) > 0
        
        # 验证数据正确性
        original_first_bar = sample_stock_data.bars[0]
        loaded_first_bar = loaded_data.bars[0]
        assert loaded_first_bar.date == original_first_bar.date
        assert loaded_first_bar.open == original_first_bar.open
        assert loaded_first_bar.close == original_first_bar.close
    
    def test_multi_year_sharding(self, parquet_storage, multi_year_stock_data):
        """测试多年数据分片存储"""
        # 保存跨年数据
        result = parquet_storage.save_stock_data(multi_year_stock_data, update_mode="overwrite")
        assert result is True
        
        # 验证分片文件存在
        base_path = Path(parquet_storage.base_path)
        file_2022 = base_path / "2022" / "000002" / "daily.parquet"
        file_2023 = base_path / "2023" / "000002" / "daily.parquet"
        
        assert file_2022.exists()
        assert file_2023.exists()
        
        # 加载全部数据
        loaded_data = parquet_storage.load_stock_data("000002")
        assert loaded_data is not None
        assert len(loaded_data.bars) > 0
        
        # 验证数据按年份正确分片
        df_2022 = pd.read_parquet(file_2022)
        df_2023 = pd.read_parquet(file_2023)
        
        assert all(pd.to_datetime(df_2022['date']).dt.year == 2022)
        assert all(pd.to_datetime(df_2023['date']).dt.year == 2023)
    
    def test_date_range_query(self, parquet_storage, multi_year_stock_data):
        """测试日期范围查询"""
        # 保存数据
        parquet_storage.save_stock_data(multi_year_stock_data, update_mode="overwrite")
        
        # 测试只查询2022年数据
        start_date = date(2022, 1, 1)
        end_date = date(2022, 12, 31)
        
        loaded_data = parquet_storage.load_stock_data("000002", start_date, end_date)
        assert loaded_data is not None
        
        # 验证日期范围
        for bar in loaded_data.bars:
            assert start_date <= bar.date <= end_date
    
    def test_incremental_update_append(self, parquet_storage, sample_stock_data):
        """测试增量更新（追加模式）"""
        # 保存初始数据
        parquet_storage.save_stock_data(sample_stock_data, update_mode="overwrite")
        
        # 创建新的数据用于追加
        new_bars = []
        last_date = sample_stock_data.bars[-1].date
        for i in range(5):
            new_date = last_date + timedelta(days=i + 1)
            # 跳过周末
            if new_date.weekday() >= 5:
                continue
            bar = StockDailyBar(
                date=new_date,
                open=200.0 + i,
                close=200.5 + i,
                high=201.0 + i,
                low=199.5 + i,
                volume=2000000 + i * 1000,
                amount=200000000 + i * 100000,
                adjust_factor=1.0
            )
            new_bars.append(bar)
        
        new_stock_data = StockData(
            symbol="000001",
            name="平安银行",
            bars=new_bars
        )
        
        # 追加数据
        original_count = len(sample_stock_data.bars)
        result = parquet_storage.save_stock_data(new_stock_data, update_mode="append")
        assert result is True
        
        # 验证数据增加
        loaded_data = parquet_storage.load_stock_data("000001")
        assert loaded_data is not None
        assert len(loaded_data.bars) > original_count
    
    def test_incremental_update_merge(self, parquet_storage, sample_stock_data):
        """测试增量更新（合并模式）"""
        # 保存初始数据
        parquet_storage.save_stock_data(sample_stock_data, update_mode="overwrite")
        
        # 创建包含重复日期的数据
        duplicate_bars = []
        first_date = sample_stock_data.bars[0].date
        
        # 使用相同日期但不同价格
        bar = StockDailyBar(
            date=first_date,
            open=150.0,  # 不同的价格
            close=150.5,
            high=151.0,
            low=149.5,
            volume=1500000,
            amount=150000000,
            adjust_factor=1.0
        )
        duplicate_bars.append(bar)
        
        duplicate_stock_data = StockData(
            symbol="000001",
            name="平安银行",
            bars=duplicate_bars
        )
        
        # 合并数据
        original_count = len(sample_stock_data.bars)
        result = parquet_storage.save_stock_data(duplicate_stock_data, update_mode="merge")
        assert result is True
        
        # 验证数据被正确合并（重复日期被覆盖）
        loaded_data = parquet_storage.load_stock_data("000001")
        assert loaded_data is not None
        assert len(loaded_data.bars) == original_count  # 数量不变
        
        # 验证重复日期的数据被更新
        first_bar = loaded_data.bars[0]
        assert first_bar.date == first_date
        assert first_bar.open == 150.0  # 新的价格
    
    def test_concurrent_read_write(self, parquet_storage, sample_stock_data):
        """测试并发读写"""
        # 保存初始数据
        parquet_storage.save_stock_data(sample_stock_data, update_mode="overwrite")
        
        results = []
        
        def read_data():
            """读取数据的线程函数"""
            try:
                data = parquet_storage.load_stock_data("000001")
                results.append(("read", data is not None))
            except Exception as e:
                results.append(("read", False, str(e)))
        
        def write_data():
            """写入数据的线程函数"""
            try:
                # 创建新数据
                new_bars = [StockDailyBar(
                    date=date(2023, 6, 1),
                    open=300.0,
                    close=300.5,
                    high=301.0,
                    low=299.5,
                    volume=3000000,
                    amount=300000000,
                    adjust_factor=1.0
                )]
                
                new_data = StockData(symbol="000001", name="平安银行", bars=new_bars)
                result = parquet_storage.save_stock_data(new_data, update_mode="append")
                results.append(("write", result))
            except Exception as e:
                results.append(("write", False, str(e)))
        
        # 启动并发线程
        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=read_data)
            t2 = threading.Thread(target=write_data)
            threads.extend([t1, t2])
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证所有操作都成功
        assert len(results) > 0
        for result in results:
            if len(result) == 2:
                assert result[1] is True
            else:
                pytest.fail(f"Operation failed: {result}")
    
    def test_data_integrity_validation(self, parquet_storage, temp_storage_path):
        """测试数据完整性校验"""
        # 创建有问题的数据
        invalid_bars = [
            StockDailyBar(
                date=date(2023, 1, 1),
                open=100.0,
                close=99.0,
                high=98.0,  # 高价小于开盘价，不合理
                low=101.0,  # 低价大于开盘价，不合理
                volume=1000000,
                amount=100000000,
                adjust_factor=1.0
            )
        ]
        
        invalid_data = StockData(
            symbol="INVALID",
            name="无效数据",
            bars=invalid_bars
        )
        
        # 保存数据
        parquet_storage.save_stock_data(invalid_data, update_mode="overwrite")
        
        # 直接验证文件完整性
        file_path = Path(temp_storage_path) / "2023" / "INVALID" / "daily.parquet"
        integrity_result = parquet_storage._verify_file_integrity(file_path)
        
        # 预期完整性检查失败
        assert integrity_result is False
    
    def test_storage_stats(self, parquet_storage, multi_year_stock_data):
        """测试存储统计信息"""
        # 保存数据
        parquet_storage.save_stock_data(multi_year_stock_data, update_mode="overwrite")
        
        # 获取统计信息
        stats = parquet_storage.get_storage_stats()
        
        assert stats['total_files'] > 0
        assert stats['total_size_mb'] > 0
        assert stats['symbols_count'] > 0
        assert stats['year_range'] is not None
        assert len(stats['year_range']) == 2
    
    def test_delete_symbol_data(self, parquet_storage, multi_year_stock_data):
        """测试删除股票数据"""
        # 保存数据
        parquet_storage.save_stock_data(multi_year_stock_data, update_mode="overwrite")
        
        # 验证数据存在
        loaded_data = parquet_storage.load_stock_data("000002")
        assert loaded_data is not None
        
        # 删除特定年份
        result = parquet_storage.delete_symbol_data("000002", 2022)
        assert result is True
        
        # 验证2022年数据被删除，2023年数据仍然存在
        loaded_data = parquet_storage.load_stock_data("000002", 
                                                    start_date=date(2022, 1, 1),
                                                    end_date=date(2022, 12, 31))
        assert loaded_data is None or len(loaded_data.bars) == 0
        
        loaded_data = parquet_storage.load_stock_data("000002",
                                                    start_date=date(2023, 1, 1), 
                                                    end_date=date(2023, 12, 31))
        assert loaded_data is not None
        
        # 删除所有数据
        result = parquet_storage.delete_symbol_data("000002")
        assert result is True
        
        # 验证所有数据被删除
        loaded_data = parquet_storage.load_stock_data("000002")
        assert loaded_data is None
    
    def test_list_available_symbols(self, parquet_storage, sample_stock_data, multi_year_stock_data):
        """测试列出可用股票代码"""
        # 初始状态应该为空
        symbols = parquet_storage.list_available_symbols()
        assert len(symbols) == 0
        
        # 保存两只股票的数据
        parquet_storage.save_stock_data(sample_stock_data, update_mode="overwrite")
        parquet_storage.save_stock_data(multi_year_stock_data, update_mode="overwrite")
        
        # 验证能够列出所有股票
        symbols = parquet_storage.list_available_symbols()
        assert len(symbols) == 2
        assert "000001" in symbols
        assert "000002" in symbols
        assert symbols == sorted(symbols)  # 应该按排序返回
    
    def test_compression_effectiveness(self, parquet_storage, sample_stock_data):
        """测试压缩效果"""
        # 保存数据
        parquet_storage.save_stock_data(sample_stock_data, update_mode="overwrite")
        
        # 获取存储统计
        stats = parquet_storage.get_storage_stats()
        
        # 验证压缩比合理
        assert stats['compression_ratio'] > 0
        assert stats['compression_ratio'] < 1.0
    
    def test_performance_large_dataset(self, parquet_storage):
        """测试大数据集性能"""
        # 创建大数据集（1万条记录）
        large_bars = []
        start_date = date(2020, 1, 1)
        
        for i in range(10000):
            current_date = start_date + timedelta(days=i)
            bar = StockDailyBar(
                date=current_date,
                open=100.0 + (i % 1000) * 0.01,
                close=100.1 + (i % 1000) * 0.01,
                high=100.5 + (i % 1000) * 0.01,
                low=99.5 + (i % 1000) * 0.01,
                volume=1000000 + i,
                amount=100000000 + i * 100,
                adjust_factor=1.0
            )
            large_bars.append(bar)
        
        large_dataset = StockData(
            symbol="LARGE",
            name="大数据集测试",
            bars=large_bars
        )
        
        # 测试保存性能
        start_time = time.time()
        result = parquet_storage.save_stock_data(large_dataset, update_mode="overwrite")
        save_time = time.time() - start_time
        
        assert result is True
        assert save_time < 10.0  # 应在10秒内完成
        
        # 测试加载性能
        start_time = time.time()
        loaded_data = parquet_storage.load_stock_data("LARGE")
        load_time = time.time() - start_time
        
        assert loaded_data is not None
        assert len(loaded_data.bars) == 10000
        assert load_time < 5.0  # 应在5秒内完成
    
    def test_global_instance(self, temp_storage_path):
        """测试全局实例"""
        storage1 = get_parquet_storage(temp_storage_path)
        storage2 = get_parquet_storage(temp_storage_path)
        
        # 应该是同一个实例
        assert storage1 is storage2
    
    def test_empty_data_handling(self, parquet_storage):
        """测试空数据处理"""
        # 创建空的股票数据
        empty_data = StockData(symbol="EMPTY", name="空数据", bars=[])
        
        # 保存空数据应该返回False
        result = parquet_storage.save_stock_data(empty_data, update_mode="overwrite")
        assert result is False
        
        # 加载不存在的股票应该返回None
        loaded_data = parquet_storage.load_stock_data("NONEXISTENT")
        assert loaded_data is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
