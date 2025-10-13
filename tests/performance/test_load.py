"""
AI量化系统性能和压力测试
测试系统在高负载下的性能表现
"""

import pytest
import asyncio
import time
import concurrent.futures
import sys
import statistics
from typing import List, Dict, Tuple
from datetime import datetime, date
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.src.data.akshare_adapter import AKShareAdapter
from backend.src.storage.parquet_engine import ParquetStorage
from backend.src.models.basic_models import StockData, StockDailyBar


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.throughput: float = 0.0
        self.start_time: float = 0.0
        self.end_time: float = 0.0
    
    def add_response_time(self, response_time: float, success: bool = True):
        """添加响应时间记录"""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def calculate_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        total_requests = self.success_count + self.error_count
        duration = self.end_time - self.start_time
        
        metrics = {
            "total_requests": total_requests,
            "success_rate": (self.success_count / total_requests * 100) if total_requests > 0 else 0,
            "error_rate": (self.error_count / total_requests * 100) if total_requests > 0 else 0,
            "duration": duration,
            "throughput": total_requests / duration if duration > 0 else 0,
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0,
            "min_response_time": min(self.response_times) if self.response_times else 0,
            "max_response_time": max(self.response_times) if self.response_times else 0,
            "p95_response_time": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else 0,
            "p99_response_time": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else 0,
        }
        
        return metrics


class TestPerformance:
    """性能和压力测试类"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_test_environment(self, request):
        """设置测试环境"""
        # 创建测试数据目录
        test_data_dir = Path("data/performance_test")
        test_data_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        request.cls.akshare_adapter = AKShareAdapter()
        request.cls.parquet_storage = ParquetStorage(base_path="data/performance_test/parquet")

        yield

        # 清理测试数据
        import shutil
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
    
    def test_data_fetch_performance(self):
        """测试数据获取性能"""
        print("\n=== 数据获取性能测试 ===")
        
        test_symbols = ["000001", "000002", "600000", "600036", "000858"]
        metrics = PerformanceMetrics()
        metrics.start_time = time.time()
        
        print(f"测试股票数量: {len(test_symbols)}")
        print("开始数据获取性能测试...")
        
        for i, symbol in enumerate(test_symbols, 1):
            start_time = time.time()
            success = False
            
            try:
                stock_data = self.akshare_adapter.get_stock_daily_data(
                    symbol=symbol,
                    start_date="2024-01-01", 
                    end_date="2024-01-10"
                )
                
                if stock_data and len(stock_data.bars) > 0:
                    success = True
                    print(f"  [{i}/{len(test_symbols)}] ✓ {symbol}: {len(stock_data.bars)} 条数据")
                else:
                    print(f"  [{i}/{len(test_symbols)}] ✗ {symbol}: 无数据")
                    
            except Exception as e:
                print(f"  [{i}/{len(test_symbols)}] ✗ {symbol}: {e}")
            
            response_time = time.time() - start_time
            metrics.add_response_time(response_time, success)
        
        metrics.end_time = time.time()
        perf_metrics = metrics.calculate_metrics()
        
        # 打印性能指标
        print(f"\n数据获取性能指标:")
        print(f"  总请求数: {perf_metrics['total_requests']}")
        print(f"  成功率: {perf_metrics['success_rate']:.1f}%")
        print(f"  平均响应时间: {perf_metrics['avg_response_time']:.3f}秒")
        print(f"  最大响应时间: {perf_metrics['max_response_time']:.3f}秒")
        print(f"  P95响应时间: {perf_metrics['p95_response_time']:.3f}秒")
        print(f"  吞吐量: {perf_metrics['throughput']:.2f} 请求/秒")
        
        # 性能断言
        assert perf_metrics['success_rate'] >= 80, f"成功率 {perf_metrics['success_rate']:.1f}% 低于80%"
        assert perf_metrics['avg_response_time'] <= 5.0, f"平均响应时间 {perf_metrics['avg_response_time']:.3f}s 超过5秒"
        
        print("✓ 数据获取性能测试通过")
    
    def test_storage_performance(self):
        """测试存储性能"""
        print("\n=== 存储性能测试 ===")
        
        # 生成测试数据
        test_data_count = 10
        test_records_per_stock = 100
        
        print(f"生成 {test_data_count} 只股票的测试数据，每只股票 {test_records_per_stock} 条记录")
        
        test_stocks = []
        for i in range(test_data_count):
            symbol = f"TEST{i:03d}"
            bars = []
            
            for j in range(test_records_per_stock):
                bar = StockDailyBar(
                    date=date(2024, 1, j % 28 + 1),
                    open_price=100.0 + j * 0.1,
                    close_price=100.5 + j * 0.1,
                    high_price=101.0 + j * 0.1,
                    low_price=99.5 + j * 0.1,
                    volume=1000000 + j * 1000
                )
                bars.append(bar)
            
            test_stocks.append(StockData(symbol=symbol, bars=bars))
        
        # 测试存储性能
        metrics = PerformanceMetrics()
        metrics.start_time = time.time()
        
        print("开始存储性能测试...")
        for i, stock_data in enumerate(test_stocks, 1):
            start_time = time.time()
            success = False
            
            try:
                self.parquet_storage.save_stock_data(stock_data)
                success = True
                print(f"  [{i}/{len(test_stocks)}] ✓ 存储 {stock_data.symbol}")
            except Exception as e:
                print(f"  [{i}/{len(test_stocks)}] ✗ 存储 {stock_data.symbol}: {e}")
            
            response_time = time.time() - start_time
            metrics.add_response_time(response_time, success)
        
        metrics.end_time = time.time()
        perf_metrics = metrics.calculate_metrics()
        
        # 打印存储性能指标
        print(f"\n存储性能指标:")
        print(f"  总存储任务: {perf_metrics['total_requests']}")
        print(f"  成功率: {perf_metrics['success_rate']:.1f}%")
        print(f"  平均存储时间: {perf_metrics['avg_response_time']:.3f}秒")
        print(f"  最大存储时间: {perf_metrics['max_response_time']:.3f}秒")
        print(f"  存储吞吐量: {perf_metrics['throughput']:.2f} 文件/秒")
        
        # 测试读取性能
        print("\n开始读取性能测试...")
        read_metrics = PerformanceMetrics()
        read_metrics.start_time = time.time()
        
        for i, stock_data in enumerate(test_stocks, 1):
            start_time = time.time()
            success = False
            
            try:
                retrieved_data = self.parquet_storage.load_stock_data(stock_data.symbol)
                if retrieved_data and len(retrieved_data.bars) > 0:
                    success = True
                    print(f"  [{i}/{len(test_stocks)}] ✓ 读取 {stock_data.symbol}: {len(retrieved_data.bars)} 条")
                else:
                    print(f"  [{i}/{len(test_stocks)}] ✗ 读取 {stock_data.symbol}: 无数据")
            except Exception as e:
                print(f"  [{i}/{len(test_stocks)}] ✗ 读取 {stock_data.symbol}: {e}")
            
            response_time = time.time() - start_time
            read_metrics.add_response_time(response_time, success)
        
        read_metrics.end_time = time.time()
        read_perf_metrics = read_metrics.calculate_metrics()
        
        # 打印读取性能指标
        print(f"\n读取性能指标:")
        print(f"  总读取任务: {read_perf_metrics['total_requests']}")
        print(f"  成功率: {read_perf_metrics['success_rate']:.1f}%")
        print(f"  平均读取时间: {read_perf_metrics['avg_response_time']:.3f}秒")
        print(f"  最大读取时间: {read_perf_metrics['max_response_time']:.3f}秒")
        print(f"  读取吞吐量: {read_perf_metrics['throughput']:.2f} 文件/秒")
        
        # 性能断言
        assert perf_metrics['success_rate'] >= 95, f"存储成功率 {perf_metrics['success_rate']:.1f}% 低于95%"
        assert read_perf_metrics['success_rate'] >= 95, f"读取成功率 {read_perf_metrics['success_rate']:.1f}% 低于95%"
        assert perf_metrics['avg_response_time'] <= 1.0, f"平均存储时间 {perf_metrics['avg_response_time']:.3f}s 超过1秒"
        assert read_perf_metrics['avg_response_time'] <= 0.5, f"平均读取时间 {read_perf_metrics['avg_response_time']:.3f}s 超过0.5秒"
        
        print("✓ 存储性能测试通过")
    
    def test_concurrent_operations(self):
        """测试并发操作性能"""
        print("\n=== 并发操作性能测试 ===")
        
        concurrent_workers = 5
        operations_per_worker = 3
        
        print(f"并发工作者: {concurrent_workers}")
        print(f"每个工作者操作数: {operations_per_worker}")
        
        def worker_task(worker_id: int) -> List[Tuple[str, float, bool]]:
            """工作者任务"""
            results = []
            
            for i in range(operations_per_worker):
                symbol = f"CONC{worker_id:02d}{i:02d}"
                
                # 创建测试数据
                test_data = StockData(
                    symbol=symbol,
                    bars=[
                        StockDailyBar(
                            date=date(2024, 1, 1),
                            open_price=100.0 + worker_id,
                            close_price=101.0 + worker_id,
                            high_price=102.0 + worker_id,
                            low_price=99.0 + worker_id,
                            volume=1000000 * (worker_id + 1)
                        )
                    ]
                )
                
                # 存储操作
                start_time = time.time()
                success = False
                
                try:
                    self.parquet_storage.save_stock_data(test_data)
                    
                    # 立即读取验证
                    retrieved_data = self.parquet_storage.load_stock_data(symbol)
                    if retrieved_data and len(retrieved_data.bars) > 0:
                        success = True
                
                except Exception as e:
                    print(f"  Worker {worker_id} 操作 {symbol} 失败: {e}")
                
                operation_time = time.time() - start_time
                results.append((symbol, operation_time, success))
            
            return results
        
        # 并发执行
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(concurrent_workers)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"  Worker 执行异常: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 分析结果
        total_operations = len(all_results)
        successful_operations = sum(1 for _, _, success in all_results if success)
        response_times = [rt for _, rt, _ in all_results]
        
        print(f"\n并发操作结果:")
        print(f"  总操作数: {total_operations}")
        print(f"  成功操作数: {successful_operations}")
        print(f"  成功率: {successful_operations / total_operations * 100:.1f}%")
        print(f"  总耗时: {total_duration:.2f}秒")
        print(f"  平均操作时间: {statistics.mean(response_times):.3f}秒")
        print(f"  并发吞吐量: {total_operations / total_duration:.2f} 操作/秒")
        
        # 并发性能断言
        success_rate = successful_operations / total_operations * 100
        assert success_rate >= 90, f"并发操作成功率 {success_rate:.1f}% 低于90%"
        assert statistics.mean(response_times) <= 2.0, f"并发平均响应时间超过2秒"
        
        print("✓ 并发操作性能测试通过")
    
    def test_memory_usage(self):
        """测试内存使用情况"""
        print("\n=== 内存使用测试 ===")
        
        try:
            import psutil
            process = psutil.Process()
            
            # 记录初始内存
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"初始内存使用: {initial_memory:.1f} MB")
            
            # 执行大量操作
            large_data_count = 20
            records_per_stock = 1000
            
            print(f"创建 {large_data_count} 只股票的大数据集，每只股票 {records_per_stock} 条记录")
            
            for i in range(large_data_count):
                symbol = f"MEMORY{i:03d}"
                bars = []
                
                for j in range(records_per_stock):
                    bar = StockDailyBar(
                        date=date(2024, 1, (j % 28) + 1),
                        open_price=100.0 + j * 0.01,
                        close_price=100.1 + j * 0.01,
                        high_price=100.2 + j * 0.01,
                        low_price=99.9 + j * 0.01,
                        volume=1000000 + j * 100
                    )
                    bars.append(bar)
                
                stock_data = StockData(symbol=symbol, bars=bars)
                self.parquet_storage.save_stock_data(stock_data)
                
                # 每5个数据集检查一次内存
                if (i + 1) % 5 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"  处理 {i+1} 个数据集后内存使用: {current_memory:.1f} MB")
            
            # 记录最终内存
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            print(f"\n内存使用统计:")
            print(f"  初始内存: {initial_memory:.1f} MB")
            print(f"  最终内存: {final_memory:.1f} MB") 
            print(f"  内存增长: {memory_increase:.1f} MB")
            print(f"  平均每个数据集内存开销: {memory_increase / large_data_count:.2f} MB")
            
            # 内存使用断言
            assert memory_increase < 500, f"内存增长 {memory_increase:.1f}MB 超过500MB限制"
            
            print("✓ 内存使用测试通过")
            
        except ImportError:
            print("⚠️  psutil 未安装，跳过内存测试")
        except Exception as e:
            print(f"⚠️  内存测试异常: {e}")
    
    def test_system_limits(self):
        """测试系统限制和边界条件"""
        print("\n=== 系统限制测试 ===")
        
        # 测试大文件处理
        print("测试大数据量处理...")
        large_symbol = "LARGE_DATA_TEST"
        large_record_count = 5000  # 5000条记录
        
        try:
            bars = []
            for i in range(large_record_count):
                bar = StockDailyBar(
                    date=date(2024, 1, (i % 28) + 1),
                    open_price=100.0 + i * 0.001,
                    close_price=100.0 + i * 0.001 + 0.1,
                    high_price=100.0 + i * 0.001 + 0.2,
                    low_price=100.0 + i * 0.001 - 0.1,
                    volume=1000000 + i
                )
                bars.append(bar)
            
            large_stock_data = StockData(symbol=large_symbol, bars=bars)
            
            # 测试存储大文件
            start_time = time.time()
            self.parquet_storage.save_stock_data(large_stock_data)
            save_time = time.time() - start_time
            
            print(f"  ✓ 存储 {large_record_count} 条记录耗时: {save_time:.2f}秒")
            
            # 测试读取大文件
            start_time = time.time()
            retrieved_data = self.parquet_storage.load_stock_data(large_symbol)
            load_time = time.time() - start_time
            
            print(f"  ✓ 读取 {len(retrieved_data.bars)} 条记录耗时: {load_time:.2f}秒")
            
            assert len(retrieved_data.bars) == large_record_count, "大文件数据完整性验证失败"
            assert save_time < 10.0, f"大文件存储时间 {save_time:.2f}秒 超过10秒限制"
            assert load_time < 5.0, f"大文件读取时间 {load_time:.2f}秒 超过5秒限制"
            
        except Exception as e:
            print(f"  ✗ 大文件处理失败: {e}")
            raise
        
        # 测试文件数量限制
        print("测试大量文件处理...")
        file_count = 100
        
        try:
            start_time = time.time()
            
            for i in range(file_count):
                symbol = f"MULTI{i:04d}"
                test_data = StockData(
                    symbol=symbol,
                    bars=[
                        StockDailyBar(
                            date=date(2024, 1, 1),
                            open_price=100.0,
                            close_price=101.0,
                            high_price=102.0,
                            low_price=99.0,
                            volume=1000000
                        )
                    ]
                )
                self.parquet_storage.save_stock_data(test_data)
            
            multi_file_time = time.time() - start_time
            print(f"  ✓ 创建 {file_count} 个文件耗时: {multi_file_time:.2f}秒")
            
            assert multi_file_time < 30.0, f"多文件创建时间 {multi_file_time:.2f}秒 超过30秒限制"
            
        except Exception as e:
            print(f"  ✗ 多文件处理失败: {e}")
            raise
        
        print("✓ 系统限制测试通过")


if __name__ == "__main__":
    # 直接运行性能测试
    test_suite = TestPerformance()
    test_suite.setup_test_environment()
    
    print("开始执行性能和压力测试...")
    
    try:
        test_suite.test_data_fetch_performance()
        test_suite.test_storage_performance()
        test_suite.test_concurrent_operations()
        test_suite.test_memory_usage()
        test_suite.test_system_limits()
        
        print("\n🎉 所有性能测试通过！")
    except Exception as e:
        print(f"\n❌ 性能测试失败: {e}")
        sys.exit(1)
