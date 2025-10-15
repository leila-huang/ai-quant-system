"""
数据服务业务逻辑

整合AKShare数据适配器和Parquet存储引擎，提供统一的数据服务接口。
"""

import uuid
import asyncio
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from loguru import logger
from pydantic import ValidationError

from backend.src.data.akshare_adapter import AKShareAdapter
from backend.src.storage.parquet_engine import get_parquet_storage
from backend.src.models.basic_models import StockData, StockDailyBar
from backend.app.core.config import settings
from backend.app.schemas.data_schemas import (
    DataSyncRequest, DataSyncResponse, SyncTaskStatus, SyncStatus,
    StockQueryRequest, StockQueryResponse, BatchStockQueryResponse,
    StockDailyData, DataStatistics, DataHealthCheck, SystemStatus,
    DataSourceStatus, PaginatedResponse, PaginationMeta
)


class DataSyncTask:
    """数据同步任务"""
    
    def __init__(self, task_id: str, request: DataSyncRequest):
        self.task_id = task_id
        self.request = request
        self.status = SyncStatus.PENDING
        self.progress = 0.0
        self.symbols_total = 0
        self.symbols_completed = 0
        self.symbols_failed = 0
        self.current_symbol = None
        self.start_time = None
        self.end_time = None
        self.error_message = None
        self.result = None
        self.failed_symbols = []


class DataService:
    """数据服务类"""
    
    def __init__(self):
        self.akshare_adapter = None
        self.parquet_storage = None
        self.sync_tasks: Dict[str, DataSyncTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        
        # 初始化存储和适配器
        self._initialize()
    
    def _initialize(self):
        """初始化数据适配器和存储引擎"""
        try:
            # 初始化AKShare适配器
            self.akshare_adapter = AKShareAdapter()
            logger.info("AKShare适配器初始化成功")
            
            # 初始化Parquet存储
            self.parquet_storage = get_parquet_storage(settings.PARQUET_STORAGE_PATH)
            logger.info("Parquet存储引擎初始化成功")
            
        except Exception as e:
            logger.error(f"数据服务初始化失败: {e}")
            raise
    
    async def create_sync_task(self, request: DataSyncRequest) -> DataSyncResponse:
        """创建数据同步任务"""
        task_id = str(uuid.uuid4())
        task = DataSyncTask(task_id, request)
        
        with self._lock:
            self.sync_tasks[task_id] = task
        
        # 确定要同步的股票列表
        # 注意：当为异步模式且未指定 symbols 时，不在此处阻塞获取全量列表，
        #       将列表获取延后至后台任务中执行，避免接口“无响应”。
        if request.symbols:
            symbols = request.symbols
        else:
            if request.async_mode:
                symbols = None  # 延迟到后台任务中获取
            else:
                # 同步模式下需要立即获取，增加超时保护
                try:
                    symbols = await self._get_all_stock_symbols()
                except Exception as e:
                    task.status = SyncStatus.FAILED
                    task.error_message = f"获取股票列表失败: {str(e)}"
                    logger.error(f"获取股票列表失败: {e}")
                    return DataSyncResponse(
                        task_id=task_id,
                        status=task.status,
                        message=task.error_message,
                        symbols_count=0
                    )
        
        task.symbols_total = len(symbols) if symbols else 0
        
        # 预估完成时间（每个股票约2秒）
        estimated_time = (len(symbols) * 2) if symbols else None
        
        # 如果是异步模式，立即返回任务ID
        if request.async_mode:
            # 提交异步任务
            asyncio.create_task(self._execute_sync_task(task, symbols))
            
            return DataSyncResponse(
                task_id=task_id,
                status=SyncStatus.PENDING,
                message="数据同步任务已创建，正在后台执行",
                symbols_count=len(symbols) if symbols else 0,
                estimated_time=estimated_time
            )
        else:
            # 同步执行
            await self._execute_sync_task(task, symbols)
            
            return DataSyncResponse(
                task_id=task_id,
                status=task.status,
                message=task.error_message or "数据同步完成",
                symbols_count=len(symbols),
                estimated_time=0
            )
    
    async def get_sync_task_status(self, task_id: str) -> Optional[SyncTaskStatus]:
        """获取同步任务状态"""
        with self._lock:
            task = self.sync_tasks.get(task_id)
            
        if not task:
            return None
            
        return SyncTaskStatus(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            symbols_total=task.symbols_total,
            symbols_completed=task.symbols_completed,
            symbols_failed=task.symbols_failed,
            current_symbol=task.current_symbol,
            start_time=task.start_time,
            end_time=task.end_time,
            error_message=task.error_message,
            result=task.result
        )
    
    async def _execute_sync_task(self, task: DataSyncTask, symbols: List[str]):
        """执行数据同步任务"""
        task.status = SyncStatus.RUNNING
        task.start_time = datetime.utcnow()
        
        try:
            # 异步模式下可能延迟获取股票列表
            if symbols is None:
                logger.info("后台任务开始获取全量股票代码列表...")
                symbols = await self._get_all_stock_symbols()
                task.symbols_total = len(symbols)
                if not symbols:
                    raise RuntimeError("无法获取股票列表或列表为空")
            
            logger.info(f"开始执行数据同步任务 {task.task_id}，股票数量: {len(symbols)}")

            success_count = 0
            failed_count = 0
            
            for i, symbol in enumerate(symbols):
                task.current_symbol = symbol
                # 在开始处理当前股票时更新一次进度，避免长时间停留在0%
                task.progress = max(task.progress, (i / len(symbols)) * 100)
                
                try:
                    # 获取股票数据
                    stock_data = await self._fetch_stock_data(
                        symbol,
                        task.request.start_date,
                        task.request.end_date,
                        task.request.data_source.value
                    )
                    
                    if stock_data and stock_data.bars:
                        # 存储数据
                        update_mode = "overwrite" if task.request.force_update else "append"
                        success = self.parquet_storage.save_stock_data(stock_data, update_mode)
                        
                        if success:
                            success_count += 1
                            task.symbols_completed += 1
                            logger.debug(f"股票 {symbol} 数据同步成功")
                        else:
                            failed_count += 1
                            task.symbols_failed += 1
                            task.failed_symbols.append(symbol)
                            logger.warning(f"股票 {symbol} 数据存储失败")
                    else:
                        failed_count += 1
                        task.symbols_failed += 1
                        task.failed_symbols.append(symbol)
                        logger.warning(f"股票 {symbol} 未获取到数据")
                
                except Exception as e:
                    failed_count += 1
                    task.symbols_failed += 1
                    task.failed_symbols.append(symbol)
                    logger.error(f"股票 {symbol} 数据同步失败: {e}")
                finally:
                    # 在每个股票处理结束后推进进度
                    task.progress = ((i + 1) / len(symbols)) * 100
                
                # 避免请求过于频繁
                await asyncio.sleep(0.1)
            
            task.progress = 100.0
            task.status = SyncStatus.SUCCESS if failed_count == 0 else SyncStatus.FAILED
            task.result = {
                "success_count": success_count,
                "failed_count": failed_count,
                "failed_symbols": task.failed_symbols[:10]  # 只记录前10个失败的
            }
            
            logger.info(f"数据同步任务 {task.task_id} 完成，成功: {success_count}，失败: {failed_count}")
            
        except Exception as e:
            task.status = SyncStatus.FAILED
            task.error_message = str(e)
            logger.error(f"数据同步任务 {task.task_id} 执行失败: {e}")
        
        finally:
            task.end_time = datetime.utcnow()
            task.current_symbol = None
    
    async def _fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        data_source: str = "akshare"
    ) -> Optional[StockData]:
        """获取股票数据"""
        try:
            if data_source == "akshare":
                # 使用线程池执行同步的数据获取操作
                loop = asyncio.get_event_loop()
                try:
                    stock_data = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor,
                            self.akshare_adapter.get_stock_data,
                            symbol,
                            start_date,
                            end_date
                        ),
                        timeout=settings.AKSHARE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(f"获取股票 {symbol} 数据超时 ({settings.AKSHARE_TIMEOUT}s)")
                    return None
                return stock_data
            else:
                logger.warning(f"不支持的数据源: {data_source}")
                return None
                
        except Exception as e:
            logger.error(f"获取股票 {symbol} 数据失败: {e}")
            return None
    
    async def _get_all_stock_symbols(self) -> List[str]:
        """获取所有股票代码"""
        try:
            loop = asyncio.get_event_loop()
            try:
                symbols = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        self.akshare_adapter.get_stock_list
                    ),
                    timeout=settings.AKSHARE_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"获取股票列表超时 ({settings.AKSHARE_TIMEOUT}s)")
                return []
            # 适配返回为字典列表的情况
            normalized = []
            for stock in symbols:
                if isinstance(stock, dict):
                    sym = stock.get('symbol')
                else:
                    sym = getattr(stock, 'symbol', None)
                if sym:
                    normalized.append(sym)
            return normalized
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    async def query_stock_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> Optional[StockQueryResponse]:
        """查询单个股票数据"""
        try:
            # 从Parquet存储查询数据
            stock_data = self.parquet_storage.load_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not stock_data or not stock_data.bars:
                return None
            
            # 应用分页
            total_count = len(stock_data.bars)
            bars = stock_data.bars[offset:offset + limit]
            
            # 转换为API响应格式
            daily_data = []
            for bar in bars:
                daily_data.append(
                    StockDailyData(
                        trade_date=bar.date,
                        open_price=bar.open_price,
                        close_price=bar.close_price,
                        high_price=bar.high_price,
                        low_price=bar.low_price,
                        volume=bar.volume,
                        amount=bar.amount,
                    )
                )
            
            return StockQueryResponse(
                symbol=symbol,
                name=stock_data.name,
                data=daily_data,
                total_count=total_count,
                has_more=offset + limit < total_count,
                next_offset=offset + limit if offset + limit < total_count else None
            )
            
        except Exception as e:
            logger.error(f"查询股票 {symbol} 数据失败: {e}")
            return None
    
    async def batch_query_stocks(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000
    ) -> BatchStockQueryResponse:
        """批量查询股票数据"""
        stocks = []
        success_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            try:
                stock_response = await self.query_stock_data(
                    symbol, start_date, end_date, limit
                )
                
                if stock_response:
                    stocks.append(stock_response)
                    success_count += 1
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                logger.error(f"批量查询股票 {symbol} 失败: {e}")
                failed_symbols.append(symbol)
        
        return BatchStockQueryResponse(
            stocks=stocks,
            total_symbols=len(symbols),
            success_count=success_count,
            failed_symbols=failed_symbols
        )
    
    async def get_available_symbols(self) -> List[str]:
        """获取可用的股票代码列表"""
        try:
            return self.parquet_storage.list_available_symbols()
        except Exception as e:
            logger.error(f"获取可用股票代码失败: {e}")
            return []
    
    async def get_data_statistics(self) -> DataStatistics:
        """获取数据统计信息"""
        try:
            stats = self.parquet_storage.get_storage_stats()
            
            return DataStatistics(
                total_symbols=stats.get("symbols_count", 0),
                total_records=stats.get("total_records", 0),
                date_range=stats.get("date_range"),
                last_update=stats.get("last_update"),
                data_sources=["akshare"],
                storage_size_mb=stats.get("total_size_mb", 0.0)
            )
            
        except Exception as e:
            logger.error(f"获取数据统计失败: {e}")
            return DataStatistics()
    
    async def health_check(self) -> DataHealthCheck:
        """数据健康检查"""
        checks = {}
        issues = []
        recommendations = []
        
        # 检查AKShare适配器
        try:
            if self.akshare_adapter:
                # 简单的连接测试
                checks["akshare_adapter"] = True
            else:
                checks["akshare_adapter"] = False
                issues.append("AKShare适配器未初始化")
        except Exception:
            checks["akshare_adapter"] = False
            issues.append("AKShare适配器检查失败")
        
        # 检查Parquet存储
        try:
            if self.parquet_storage:
                stats = self.parquet_storage.get_storage_stats()
                checks["parquet_storage"] = True
                
                # 检查数据是否过期
                if stats.get("last_update"):
                    last_update = stats["last_update"]
                    if isinstance(last_update, str):
                        last_update = datetime.fromisoformat(last_update)
                    if (datetime.utcnow() - last_update).days > 7:
                        issues.append("数据超过7天未更新")
                        recommendations.append("建议执行数据同步")
            else:
                checks["parquet_storage"] = False
                issues.append("Parquet存储未初始化")
        except Exception:
            checks["parquet_storage"] = False
            issues.append("Parquet存储检查失败")
        
        # 检查同步任务状态
        active_tasks = len([t for t in self.sync_tasks.values() if t.status == SyncStatus.RUNNING])
        checks["sync_tasks"] = active_tasks < 10  # 活跃任务不超过10个
        
        if active_tasks > 5:
            issues.append(f"活跃同步任务过多: {active_tasks}")
            recommendations.append("考虑优化同步策略或增加处理能力")
        
        # 总体状态
        status = "healthy" if all(checks.values()) and not issues else "warning" if not issues else "error"
        
        return DataHealthCheck(
            status=status,
            checks=checks,
            statistics=await self.get_data_statistics(),
            last_sync_tasks=self._get_recent_sync_tasks(),
            issues=issues,
            recommendations=recommendations
        )
    
    def _get_recent_sync_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取最近的同步任务"""
        tasks = []
        
        # 按创建时间排序，取最近的任务
        sorted_tasks = sorted(
            self.sync_tasks.values(),
            key=lambda t: t.start_time or datetime.min,
            reverse=True
        )
        
        for task in sorted_tasks[:limit]:
            tasks.append({
                "task_id": task.task_id,
                "status": task.status.value,
                "symbols_total": task.symbols_total,
                "symbols_completed": task.symbols_completed,
                "start_time": task.start_time.isoformat() if task.start_time else None,
                "end_time": task.end_time.isoformat() if task.end_time else None,
            })
        
        return tasks
    
    async def cancel_sync_task(self, task_id: str) -> bool:
        """取消同步任务"""
        with self._lock:
            task = self.sync_tasks.get(task_id)
            
        if not task:
            return False
        
        if task.status == SyncStatus.RUNNING:
            task.status = SyncStatus.CANCELLED
            task.end_time = datetime.utcnow()
            logger.info(f"同步任务 {task_id} 已取消")
            return True
        
        return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """清理旧的任务记录"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        with self._lock:
            tasks_to_remove = []
            for task_id, task in self.sync_tasks.items():
                if task.end_time and task.end_time < cutoff_time:
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.sync_tasks[task_id]
        
        if tasks_to_remove:
            logger.info(f"清理了 {len(tasks_to_remove)} 个过期的同步任务")


# 全局数据服务实例
_data_service_instance = None


def get_data_service() -> DataService:
    """获取数据服务实例（单例模式）"""
    global _data_service_instance
    
    if _data_service_instance is None:
        _data_service_instance = DataService()
    
    return _data_service_instance
