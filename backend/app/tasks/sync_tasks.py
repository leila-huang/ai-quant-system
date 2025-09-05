"""
异步数据同步任务

提供后台数据同步任务的执行和管理。
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from backend.app.core.config import settings
from backend.app.services.data_service import get_data_service
from backend.app.schemas.data_schemas import DataSyncRequest, SyncStatus


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 10


@dataclass
class AsyncTask:
    """异步任务定义"""
    task_id: str
    task_type: str
    priority: int
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: SyncStatus = SyncStatus.PENDING
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    max_retries: int = 3
    retry_count: int = 0
    
    def __post_init__(self):
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at


class TaskQueue:
    """任务队列管理"""
    
    def __init__(self, max_concurrent_tasks: int = 3):
        self.tasks: Dict[str, AsyncTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # 启动任务调度器
        self._scheduler_task = None
        self.start_scheduler()
    
    def add_task(
        self,
        task_type: str,
        priority: int = TaskPriority.NORMAL.value,
        scheduled_at: Optional[datetime] = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """添加任务到队列"""
        task_id = str(uuid.uuid4())
        
        task = AsyncTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            created_at=datetime.utcnow(),
            scheduled_at=scheduled_at or datetime.utcnow(),
            max_retries=max_retries
        )
        
        # 将任务参数存储在result字段中（临时使用）
        task.result = kwargs
        
        with self._lock:
            self.tasks[task_id] = task
        
        logger.info(f"任务 {task_id} ({task_type}) 已添加到队列，优先级: {priority}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[AsyncTask]:
        """获取任务状态"""
        with self._lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status == SyncStatus.RUNNING:
                # 取消正在运行的任务
                asyncio_task = self.running_tasks.get(task_id)
                if asyncio_task:
                    asyncio_task.cancel()
                    del self.running_tasks[task_id]
            
            task.status = SyncStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            
        logger.info(f"任务 {task_id} 已取消")
        return True
    
    def start_scheduler(self):
        """启动任务调度器"""
        if self._scheduler_task is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    def stop_scheduler(self):
        """停止任务调度器"""
        self._stop_event.set()
        
        # 取消所有运行中的任务
        for task_id, asyncio_task in self.running_tasks.items():
            asyncio_task.cancel()
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
        
        self._executor.shutdown(wait=True)
    
    async def _scheduler_loop(self):
        """任务调度循环"""
        logger.info("任务调度器已启动")
        
        while not self._stop_event.is_set():
            try:
                await self._process_pending_tasks()
                await asyncio.sleep(1)  # 每秒检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"任务调度器出错: {e}")
                await asyncio.sleep(5)
        
        logger.info("任务调度器已停止")
    
    async def _process_pending_tasks(self):
        """处理待执行任务"""
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return
        
        # 获取待执行的任务
        pending_tasks = []
        current_time = datetime.utcnow()
        
        with self._lock:
            for task in self.tasks.values():
                if (task.status == SyncStatus.PENDING and 
                    task.scheduled_at <= current_time):
                    pending_tasks.append(task)
        
        if not pending_tasks:
            return
        
        # 按优先级和创建时间排序
        pending_tasks.sort(key=lambda t: (-t.priority, t.created_at))
        
        # 启动任务（最多到并发限制）
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)
        for task in pending_tasks[:available_slots]:
            await self._start_task(task)
    
    async def _start_task(self, task: AsyncTask):
        """启动单个任务"""
        task.status = SyncStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        logger.info(f"开始执行任务 {task.task_id} ({task.task_type})")
        
        # 创建异步任务
        asyncio_task = asyncio.create_task(self._execute_task(task))
        self.running_tasks[task.task_id] = asyncio_task
        
        # 设置任务完成回调
        asyncio_task.add_done_callback(
            lambda t: self._task_completed(task.task_id, t)
        )
    
    async def _execute_task(self, task: AsyncTask):
        """执行任务"""
        try:
            if task.task_type == "data_sync":
                await self._execute_data_sync_task(task)
            elif task.task_type == "data_cleanup":
                await self._execute_cleanup_task(task)
            elif task.task_type == "health_check":
                await self._execute_health_check_task(task)
            else:
                raise ValueError(f"未知的任务类型: {task.task_type}")
                
        except Exception as e:
            logger.error(f"任务 {task.task_id} 执行失败: {e}")
            task.error = str(e)
            task.status = SyncStatus.FAILED
            
            # 重试机制
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = SyncStatus.PENDING
                task.scheduled_at = datetime.utcnow() + timedelta(minutes=2 ** task.retry_count)
                logger.info(f"任务 {task.task_id} 将在 {task.scheduled_at} 重试 (第{task.retry_count}次)")
            else:
                logger.error(f"任务 {task.task_id} 重试次数超限，标记为失败")
    
    async def _execute_data_sync_task(self, task: AsyncTask):
        """执行数据同步任务"""
        task_params = task.result  # 之前存储在result中的参数
        
        # 构造同步请求
        sync_request = DataSyncRequest(
            symbols=task_params.get("symbols"),
            start_date=task_params.get("start_date"),
            end_date=task_params.get("end_date"),
            data_source=task_params.get("data_source", "akshare"),
            force_update=task_params.get("force_update", False),
            async_mode=False,  # 在任务队列中始终同步执行
            priority=task.priority
        )
        
        # 获取数据服务并执行同步
        data_service = get_data_service()
        response = await data_service.create_sync_task(sync_request)
        
        # 等待同步完成
        while True:
            status = await data_service.get_sync_task_status(response.task_id)
            if not status:
                break
                
            task.progress = status.progress
            
            if status.status in [SyncStatus.SUCCESS, SyncStatus.FAILED, SyncStatus.CANCELLED]:
                task.result = {
                    "sync_task_id": response.task_id,
                    "symbols_completed": status.symbols_completed,
                    "symbols_failed": status.symbols_failed,
                    "result": status.result
                }
                
                if status.status == SyncStatus.SUCCESS:
                    task.status = SyncStatus.SUCCESS
                    logger.info(f"数据同步任务 {task.task_id} 完成")
                else:
                    task.status = SyncStatus.FAILED
                    task.error = status.error_message
                    logger.error(f"数据同步任务 {task.task_id} 失败: {status.error_message}")
                break
                
            await asyncio.sleep(2)
    
    async def _execute_cleanup_task(self, task: AsyncTask):
        """执行清理任务"""
        task_params = task.result
        
        # 清理旧的任务记录
        data_service = get_data_service()
        max_age_hours = task_params.get("max_age_hours", 24)
        data_service.cleanup_old_tasks(max_age_hours)
        
        task.status = SyncStatus.SUCCESS
        task.progress = 100.0
        task.result = {"cleaned_tasks": "completed"}
        
        logger.info(f"清理任务 {task.task_id} 完成")
    
    async def _execute_health_check_task(self, task: AsyncTask):
        """执行健康检查任务"""
        data_service = get_data_service()
        health_result = await data_service.health_check()
        
        task.status = SyncStatus.SUCCESS
        task.progress = 100.0
        task.result = {
            "overall_status": health_result.status,
            "issues_count": len(health_result.issues),
            "checks_passed": sum(health_result.checks.values())
        }
        
        logger.info(f"健康检查任务 {task.task_id} 完成，状态: {health_result.status}")
    
    def _task_completed(self, task_id: str, asyncio_task: asyncio.Task):
        """任务完成回调"""
        with self._lock:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            task = self.tasks.get(task_id)
            if task and task.status == SyncStatus.RUNNING:
                if asyncio_task.cancelled():
                    task.status = SyncStatus.CANCELLED
                elif asyncio_task.exception():
                    task.status = SyncStatus.FAILED
                    task.error = str(asyncio_task.exception())
                else:
                    # 状态由任务执行函数设置
                    pass
                
                task.completed_at = datetime.utcnow()
        
        if asyncio_task.exception() and not asyncio_task.cancelled():
            logger.error(f"任务 {task_id} 异常完成: {asyncio_task.exception()}")
        else:
            logger.info(f"任务 {task_id} 已完成")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        with self._lock:
            stats = {
                "total_tasks": len(self.tasks),
                "pending_tasks": len([t for t in self.tasks.values() if t.status == SyncStatus.PENDING]),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len([t for t in self.tasks.values() if t.status == SyncStatus.SUCCESS]),
                "failed_tasks": len([t for t in self.tasks.values() if t.status == SyncStatus.FAILED]),
                "cancelled_tasks": len([t for t in self.tasks.values() if t.status == SyncStatus.CANCELLED]),
                "max_concurrent": self.max_concurrent_tasks
            }
        
        return stats
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """清理已完成的任务"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        with self._lock:
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if (task.completed_at and 
                    task.completed_at < cutoff_time and 
                    task.status in [SyncStatus.SUCCESS, SyncStatus.FAILED, SyncStatus.CANCELLED]):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
        
        logger.info(f"清理了 {len(tasks_to_remove)} 个过期的已完成任务")


# 全局任务队列实例
_task_queue_instance = None


def get_task_queue() -> TaskQueue:
    """获取任务队列实例（单例模式）"""
    global _task_queue_instance
    
    if _task_queue_instance is None:
        max_concurrent = getattr(settings, 'MAX_CONCURRENT_SYNC_TASKS', 3)
        _task_queue_instance = TaskQueue(max_concurrent_tasks=max_concurrent)
    
    return _task_queue_instance


def schedule_data_sync(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    priority: int = TaskPriority.NORMAL.value,
    scheduled_at: Optional[datetime] = None
) -> str:
    """调度数据同步任务"""
    queue = get_task_queue()
    
    task_id = queue.add_task(
        task_type="data_sync",
        priority=priority,
        scheduled_at=scheduled_at,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info(f"数据同步任务 {task_id} 已调度")
    return task_id


def schedule_cleanup_task(
    max_age_hours: int = 24,
    priority: int = TaskPriority.LOW.value,
    scheduled_at: Optional[datetime] = None
) -> str:
    """调度清理任务"""
    queue = get_task_queue()
    
    task_id = queue.add_task(
        task_type="data_cleanup",
        priority=priority,
        scheduled_at=scheduled_at,
        max_age_hours=max_age_hours
    )
    
    logger.info(f"清理任务 {task_id} 已调度")
    return task_id


def schedule_health_check(
    priority: int = TaskPriority.NORMAL.value,
    scheduled_at: Optional[datetime] = None
) -> str:
    """调度健康检查任务"""
    queue = get_task_queue()
    
    task_id = queue.add_task(
        task_type="health_check",
        priority=priority,
        scheduled_at=scheduled_at
    )
    
    logger.info(f"健康检查任务 {task_id} 已调度")
    return task_id


# 定期任务调度
def setup_periodic_tasks():
    """设置定期任务"""
    import schedule
    import time
    import threading
    
    def run_periodic():
        # 每天凌晨2点执行数据同步
        schedule.every().day.at("02:00").do(schedule_data_sync)
        
        # 每天凌晨3点执行清理任务
        schedule.every().day.at("03:00").do(schedule_cleanup_task)
        
        # 每小时执行健康检查
        schedule.every().hour.do(schedule_health_check)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    # 在后台线程中运行定期任务
    threading.Thread(target=run_periodic, daemon=True).start()
    logger.info("定期任务调度已设置")
