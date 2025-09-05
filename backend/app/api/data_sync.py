"""
数据同步API路由

提供股票数据同步、查询和管理的REST API接口。
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.services.data_service import get_data_service
from backend.app.tasks.sync_tasks import get_task_queue, schedule_data_sync
from backend.app.schemas.data_schemas import (
    DataSyncRequest, DataSyncResponse, SyncTaskStatus,
    StockQueryRequest, StockQueryResponse, BatchStockQueryResponse,
    DataHealthCheck, DataStatistics, SystemStatus, PaginatedResponse
)
from backend.app.core.exceptions import (
    BusinessException, DataValidationException,
    ExternalServiceException, RateLimitException
)


router = APIRouter()


# 依赖注入
def get_data_service_dep():
    """获取数据服务依赖"""
    return get_data_service()


def get_task_queue_dep():
    """获取任务队列依赖"""
    return get_task_queue()


# 限流装饰器（简化实现）
async def rate_limit_check(request_type: str = "default"):
    """API限流检查"""
    # 在实际生产中，这里应该使用Redis等实现真正的限流
    # 这里只是一个占位符
    pass


@router.post("/sync", response_model=DataSyncResponse, summary="启动数据同步")
async def create_sync_task(
    request: DataSyncRequest,
    background_tasks: BackgroundTasks,
    data_service = Depends(get_data_service_dep)
):
    """
    启动股票数据同步任务
    
    - **symbols**: 股票代码列表，为空则同步全部
    - **start_date**: 开始日期，默认为30天前
    - **end_date**: 结束日期，默认为今日
    - **data_source**: 数据源，支持 akshare
    - **force_update**: 是否强制更新已存在的数据
    - **async_mode**: 是否异步执行
    - **priority**: 任务优先级（1-10）
    """
    try:
        await rate_limit_check("data_sync")
        
        # 参数验证
        if request.symbols and len(request.symbols) > 100:
            raise DataValidationException("单次同步股票数量不能超过100个")
        
        if request.start_date and request.end_date:
            date_diff = (request.end_date - request.start_date).days
            if date_diff > 365:
                raise DataValidationException("同步日期范围不能超过365天")
        
        # 设置默认日期范围
        if not request.start_date:
            request.start_date = date.today() - timedelta(days=30)
        if not request.end_date:
            request.end_date = date.today()
        
        # 执行同步任务
        response = await data_service.create_sync_task(request)
        
        # 如果是异步模式，返回任务状态
        if request.async_mode:
            background_tasks.add_task(
                _monitor_sync_task,
                response.task_id,
                data_service
            )
        
        return response
        
    except DataValidationException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建同步任务失败: {str(e)}"
        )


async def _monitor_sync_task(task_id: str, data_service):
    """监控同步任务执行（后台任务）"""
    # 这是一个后台任务，用于监控长时间运行的同步任务
    # 在实际生产中，可以添加进度通知、错误报告等功能
    pass


@router.get("/sync/{task_id}/status", response_model=SyncTaskStatus, summary="查询同步任务状态")
async def get_sync_task_status(
    task_id: str = Path(..., description="同步任务ID"),
    data_service = Depends(get_data_service_dep)
):
    """
    查询数据同步任务的执行状态
    
    返回任务的详细执行状态，包括进度、已完成数量、错误信息等。
    """
    try:
        task_status = await data_service.get_sync_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到任务ID {task_id}"
            )
        
        return task_status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询任务状态失败: {str(e)}"
        )


@router.delete("/sync/{task_id}", summary="取消同步任务")
async def cancel_sync_task(
    task_id: str = Path(..., description="同步任务ID"),
    data_service = Depends(get_data_service_dep)
):
    """
    取消正在执行或等待中的数据同步任务
    """
    try:
        success = await data_service.cancel_sync_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务 {task_id} 不存在或无法取消"
            )
        
        return {"message": "任务已取消", "task_id": task_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取消任务失败: {str(e)}"
        )


@router.get("/stocks/{symbol}", response_model=StockQueryResponse, summary="查询单个股票数据")
async def get_stock_data(
    symbol: str = Path(..., description="股票代码", example="000001"),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    limit: int = Query(1000, ge=1, le=10000, description="最大返回记录数"),
    offset: int = Query(0, ge=0, description="偏移量"),
    data_service = Depends(get_data_service_dep)
):
    """
    查询指定股票的历史数据
    
    - **symbol**: 股票代码，如 000001
    - **start_date**: 开始日期，格式 YYYY-MM-DD
    - **end_date**: 结束日期，格式 YYYY-MM-DD
    - **limit**: 最大返回记录数，默认1000
    - **offset**: 分页偏移量，默认0
    """
    try:
        await rate_limit_check("stock_query")
        
        stock_data = await data_service.query_stock_data(
            symbol=symbol.upper(),
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        if not stock_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到股票 {symbol} 的数据"
            )
        
        return stock_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询股票数据失败: {str(e)}"
        )


@router.post("/stocks/batch", response_model=BatchStockQueryResponse, summary="批量查询股票数据")
async def batch_query_stocks(
    symbols: List[str],
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    limit: int = Query(500, ge=1, le=5000, description="每个股票的最大记录数"),
    data_service = Depends(get_data_service_dep)
):
    """
    批量查询多个股票的历史数据
    
    - **symbols**: 股票代码列表，最多50个
    - **start_date**: 开始日期
    - **end_date**: 结束日期
    - **limit**: 每个股票的最大记录数
    """
    try:
        await rate_limit_check("batch_query")
        
        # 限制批量查询数量
        if len(symbols) > 50:
            raise DataValidationException("批量查询股票数量不能超过50个")
        
        # 标准化股票代码
        symbols = [symbol.upper() for symbol in symbols]
        
        batch_result = await data_service.batch_query_stocks(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return batch_result
        
    except DataValidationException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量查询失败: {str(e)}"
        )


@router.get("/symbols", summary="获取可用股票代码")
async def get_available_symbols(
    limit: int = Query(1000, ge=1, le=10000, description="最大返回数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    pattern: Optional[str] = Query(None, description="股票代码过滤模式"),
    data_service = Depends(get_data_service_dep)
):
    """
    获取系统中可用的股票代码列表
    
    - **limit**: 最大返回数量
    - **offset**: 分页偏移量
    - **pattern**: 股票代码过滤模式，如 "00" 匹配以00开头的股票
    """
    try:
        symbols = await data_service.get_available_symbols()
        
        # 应用过滤模式
        if pattern:
            symbols = [s for s in symbols if pattern in s]
        
        # 应用分页
        total_count = len(symbols)
        paginated_symbols = symbols[offset:offset + limit]
        
        return {
            "symbols": paginated_symbols,
            "total_count": total_count,
            "has_more": offset + limit < total_count,
            "next_offset": offset + limit if offset + limit < total_count else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取股票代码列表失败: {str(e)}"
        )


@router.get("/statistics", response_model=DataStatistics, summary="数据统计信息")
async def get_data_statistics(
    data_service = Depends(get_data_service_dep)
):
    """
    获取数据统计信息
    
    包括股票数量、记录总数、数据日期范围、存储大小等统计信息。
    """
    try:
        stats = await data_service.get_data_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据统计失败: {str(e)}"
        )


@router.get("/health", response_model=DataHealthCheck, summary="数据系统健康检查")
async def data_health_check(
    data_service = Depends(get_data_service_dep)
):
    """
    数据系统健康检查
    
    检查数据源状态、存储状态、同步任务状态等，并提供改进建议。
    """
    try:
        health_check = await data_service.health_check()
        return health_check
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"健康检查失败: {str(e)}"
        )


@router.get("/tasks", summary="查询任务队列状态")
async def get_task_queue_status(
    task_queue = Depends(get_task_queue_dep)
):
    """
    查询后台任务队列状态
    
    显示当前队列中的任务统计信息。
    """
    try:
        stats = task_queue.get_queue_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务队列状态失败: {str(e)}"
        )


@router.post("/tasks/schedule", summary="调度后台任务")
async def schedule_background_task(
    task_type: str = Query(..., description="任务类型: data_sync, data_cleanup, health_check"),
    priority: int = Query(5, ge=1, le=10, description="任务优先级"),
    symbols: Optional[List[str]] = Query(None, description="股票代码列表（仅data_sync任务）"),
    scheduled_at: Optional[datetime] = Query(None, description="调度时间"),
    task_queue = Depends(get_task_queue_dep)
):
    """
    调度后台任务
    
    支持调度数据同步、数据清理、健康检查等后台任务。
    """
    try:
        if task_type not in ["data_sync", "data_cleanup", "health_check"]:
            raise DataValidationException(f"不支持的任务类型: {task_type}")
        
        task_params = {}
        if task_type == "data_sync" and symbols:
            task_params["symbols"] = symbols
        
        task_id = task_queue.add_task(
            task_type=task_type,
            priority=priority,
            scheduled_at=scheduled_at,
            **task_params
        )
        
        return {
            "message": "任务已调度",
            "task_id": task_id,
            "task_type": task_type,
            "priority": priority,
            "scheduled_at": scheduled_at or datetime.utcnow()
        }
        
    except DataValidationException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"调度任务失败: {str(e)}"
        )


@router.delete("/tasks/{task_id}", summary="取消后台任务")
async def cancel_background_task(
    task_id: str = Path(..., description="任务ID"),
    task_queue = Depends(get_task_queue_dep)
):
    """
    取消指定的后台任务
    """
    try:
        success = task_queue.cancel_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务 {task_id} 不存在或无法取消"
            )
        
        return {"message": "任务已取消", "task_id": task_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取消任务失败: {str(e)}"
        )


@router.post("/maintenance/cleanup", summary="清理维护")
async def maintenance_cleanup(
    max_age_hours: int = Query(24, ge=1, le=168, description="清理超过指定小时数的记录"),
    cleanup_tasks: bool = Query(True, description="是否清理已完成的任务"),
    cleanup_logs: bool = Query(False, description="是否清理日志文件"),
    data_service = Depends(get_data_service_dep),
    task_queue = Depends(get_task_queue_dep)
):
    """
    执行系统清理维护
    
    清理过期的任务记录、日志文件等。
    """
    try:
        results = {}
        
        if cleanup_tasks:
            # 清理数据服务中的旧任务
            data_service.cleanup_old_tasks(max_age_hours)
            
            # 清理任务队列中的已完成任务
            task_queue.cleanup_completed_tasks(max_age_hours)
            
            results["tasks_cleaned"] = True
        
        if cleanup_logs:
            # 日志清理逻辑（这里只是示例）
            results["logs_cleaned"] = True
        
        return {
            "message": "清理维护完成",
            "results": results,
            "cleanup_time": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清理维护失败: {str(e)}"
        )


# 开发和测试相关的端点
if settings.DEBUG:
    @router.post("/dev/test-data", summary="创建测试数据")
    async def create_test_data(
        symbol: str = Query("TEST001", description="测试股票代码"),
        days: int = Query(30, ge=1, le=365, description="生成天数"),
        data_service = Depends(get_data_service_dep)
    ):
        """
        创建测试数据（仅开发模式）
        
        生成指定天数的随机股票数据，用于开发和测试。
        """
        try:
            from backend.src.models.basic_models import StockDailyBar, StockData
            import random
            
            # 生成测试数据
            bars = []
            base_price = 100.0
            current_date = date.today()
            
            for i in range(days):
                # 简单的随机价格生成
                change = random.uniform(-0.05, 0.05)
                base_price = base_price * (1 + change)
                
                high_price = base_price * (1 + random.uniform(0, 0.02))
                low_price = base_price * (1 - random.uniform(0, 0.02))
                
                bar = StockDailyBar(
                    date=current_date - timedelta(days=days-1-i),
                    open_price=round(base_price, 2),
                    close_price=round(base_price, 2),
                    high_price=round(high_price, 2),
                    low_price=round(low_price, 2),
                    volume=random.randint(100000, 1000000),
                    amount=random.randint(10000000, 100000000)
                )
                bars.append(bar)
            
            # 创建股票数据
            stock_data = StockData(
                symbol=symbol,
                name=f"测试股票{symbol}",
                bars=bars
            )
            
            # 保存到存储
            parquet_storage = data_service.parquet_storage
            success = parquet_storage.save_stock_data(stock_data, update_mode="overwrite")
            
            if success:
                return {
                    "message": f"成功创建测试数据 {symbol}",
                    "symbol": symbol,
                    "bars_count": len(bars),
                    "date_range": f"{bars[0].date} 到 {bars[-1].date}"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="保存测试数据失败"
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"创建测试数据失败: {str(e)}"
            )
