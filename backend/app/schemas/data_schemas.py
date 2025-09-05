"""
数据API的Pydantic模型

定义数据同步、查询等API的请求和响应模型。
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class SyncStatus(str, Enum):
    """数据同步状态枚举"""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SyncDataSource(str, Enum):
    """数据源枚举"""
    AKSHARE = "akshare"
    EASTMONEY = "eastmoney"
    MANUAL = "manual"


class StockCodeFormat(str, Enum):
    """股票代码格式"""
    SIMPLE = "simple"  # 如: 000001
    FULL = "full"      # 如: 000001.SZ
    AUTO = "auto"      # 自动检测


class DataSyncRequest(BaseModel):
    """数据同步请求模型"""
    symbols: Optional[List[str]] = Field(None, description="股票代码列表，为空则同步全部")
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期，默认为今日")
    data_source: SyncDataSource = Field(SyncDataSource.AKSHARE, description="数据源")
    force_update: bool = Field(False, description="是否强制更新已存在的数据")
    async_mode: bool = Field(True, description="是否异步执行")
    priority: int = Field(5, ge=1, le=10, description="任务优先级（1-10）")
    
    @validator("end_date")
    def validate_end_date(cls, v, values):
        """验证结束日期"""
        if v and "start_date" in values and values["start_date"]:
            if v < values["start_date"]:
                raise ValueError("结束日期不能早于开始日期")
        return v
    
    @validator("symbols")
    def validate_symbols(cls, v):
        """验证股票代码格式"""
        if v:
            for symbol in v:
                if not symbol or len(symbol) < 6:
                    raise ValueError(f"无效的股票代码格式: {symbol}")
        return v


class DataSyncResponse(BaseModel):
    """数据同步响应模型"""
    task_id: str = Field(..., description="同步任务ID")
    status: SyncStatus = Field(..., description="同步状态")
    message: str = Field(..., description="状态信息")
    symbols_count: int = Field(0, description="待同步股票数量")
    estimated_time: Optional[int] = Field(None, description="预估完成时间（秒）")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class SyncTaskStatus(BaseModel):
    """同步任务状态模型"""
    task_id: str = Field(..., description="任务ID")
    status: SyncStatus = Field(..., description="任务状态")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="完成进度百分比")
    symbols_total: int = Field(0, description="总股票数量")
    symbols_completed: int = Field(0, description="已完成数量")
    symbols_failed: int = Field(0, description="失败数量")
    current_symbol: Optional[str] = Field(None, description="当前处理的股票")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    error_message: Optional[str] = Field(None, description="错误信息")
    result: Optional[Dict[str, Any]] = Field(None, description="执行结果")


class StockQueryRequest(BaseModel):
    """股票数据查询请求"""
    symbols: Optional[List[str]] = Field(None, description="股票代码列表")
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    limit: int = Field(1000, ge=1, le=10000, description="最大返回记录数")
    offset: int = Field(0, ge=0, description="偏移量")
    sort_by: str = Field("date", description="排序字段")
    sort_order: str = Field("desc", description="排序方式: asc, desc")
    include_volume: bool = Field(True, description="是否包含成交量数据")
    include_amount: bool = Field(True, description="是否包含成交额数据")
    
    @validator("sort_by")
    def validate_sort_by(cls, v):
        """验证排序字段"""
        allowed_fields = ["date", "open_price", "close_price", "high_price", "low_price", "volume"]
        if v not in allowed_fields:
            raise ValueError(f"排序字段必须是以下之一: {allowed_fields}")
        return v
    
    @validator("sort_order")
    def validate_sort_order(cls, v):
        """验证排序方式"""
        if v.lower() not in ["asc", "desc"]:
            raise ValueError("排序方式必须是 'asc' 或 'desc'")
        return v.lower()


class StockDailyData(BaseModel):
    """股票日线数据模型"""
    date: date = Field(..., description="交易日期")
    open_price: float = Field(..., description="开盘价")
    close_price: float = Field(..., description="收盘价")
    high_price: float = Field(..., description="最高价")
    low_price: float = Field(..., description="最低价")
    volume: Optional[float] = Field(None, description="成交量")
    amount: Optional[float] = Field(None, description="成交额")
    pct_change: Optional[float] = Field(None, description="涨跌幅")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


class StockQueryResponse(BaseModel):
    """股票数据查询响应"""
    symbol: str = Field(..., description="股票代码")
    name: Optional[str] = Field(None, description="股票名称")
    data: List[StockDailyData] = Field(..., description="日线数据列表")
    total_count: int = Field(..., description="总记录数")
    has_more: bool = Field(False, description="是否有更多数据")
    next_offset: Optional[int] = Field(None, description="下一页偏移量")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据信息")


class BatchStockQueryResponse(BaseModel):
    """批量股票查询响应"""
    stocks: List[StockQueryResponse] = Field(..., description="股票数据列表")
    total_symbols: int = Field(..., description="查询的股票总数")
    success_count: int = Field(..., description="成功获取数据的股票数")
    failed_symbols: List[str] = Field(default_factory=list, description="获取失败的股票代码")
    query_time: datetime = Field(default_factory=datetime.utcnow, description="查询时间")


class DataStatistics(BaseModel):
    """数据统计模型"""
    total_symbols: int = Field(0, description="总股票数量")
    total_records: int = Field(0, description="总记录数")
    date_range: Optional[Dict[str, str]] = Field(None, description="数据日期范围")
    last_update: Optional[datetime] = Field(None, description="最后更新时间")
    data_sources: List[str] = Field(default_factory=list, description="数据源列表")
    storage_size_mb: float = Field(0.0, description="存储大小(MB)")


class DataHealthCheck(BaseModel):
    """数据健康检查模型"""
    status: str = Field(..., description="健康状态")
    checks: Dict[str, bool] = Field(..., description="各项检查结果")
    statistics: DataStatistics = Field(..., description="数据统计")
    last_sync_tasks: List[Dict[str, Any]] = Field(default_factory=list, description="最近的同步任务")
    issues: List[str] = Field(default_factory=list, description="发现的问题")
    recommendations: List[str] = Field(default_factory=list, description="改进建议")


class CacheConfig(BaseModel):
    """缓存配置模型"""
    enabled: bool = Field(True, description="是否启用缓存")
    ttl: int = Field(300, description="缓存过期时间（秒）")
    max_size: int = Field(1000, description="最大缓存条目数")
    key_prefix: str = Field("stock_data", description="缓存键前缀")


class APIRateLimit(BaseModel):
    """API限流配置"""
    enabled: bool = Field(True, description="是否启用限流")
    requests_per_minute: int = Field(100, description="每分钟请求数限制")
    burst_size: int = Field(20, description="突发请求大小")
    
    
class DataSourceStatus(BaseModel):
    """数据源状态模型"""
    name: str = Field(..., description="数据源名称")
    available: bool = Field(..., description="是否可用")
    last_check: datetime = Field(..., description="最后检查时间")
    response_time: Optional[float] = Field(None, description="响应时间(秒)")
    error_message: Optional[str] = Field(None, description="错误信息")
    success_rate: float = Field(0.0, description="成功率")
    
    
class SystemStatus(BaseModel):
    """系统状态模型"""
    overall_status: str = Field(..., description="总体状态")
    data_sources: List[DataSourceStatus] = Field(..., description="数据源状态")
    storage_status: Dict[str, Any] = Field(..., description="存储状态")
    active_sync_tasks: int = Field(0, description="活跃同步任务数")
    queue_size: int = Field(0, description="任务队列大小")
    last_successful_sync: Optional[datetime] = Field(None, description="最后成功同步时间")


# 分页相关模型
class PaginationMeta(BaseModel):
    """分页元数据"""
    current_page: int = Field(..., description="当前页码")
    total_pages: int = Field(..., description="总页数")
    page_size: int = Field(..., description="每页大小")
    total_items: int = Field(..., description="总条目数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")


class PaginatedResponse(BaseModel):
    """通用分页响应"""
    data: List[Any] = Field(..., description="数据列表")
    pagination: PaginationMeta = Field(..., description="分页信息")
    

# 错误响应模型
class ErrorDetail(BaseModel):
    """错误详情"""
    code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误信息")
    field: Optional[str] = Field(None, description="相关字段")
    

class ErrorResponse(BaseModel):
    """标准错误响应"""
    success: bool = Field(False, description="是否成功")
    error: ErrorDetail = Field(..., description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    request_id: Optional[str] = Field(None, description="请求ID")
