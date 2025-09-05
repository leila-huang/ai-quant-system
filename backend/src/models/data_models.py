"""
数据模型定义 - Pydantic模型
定义系统中使用的各种数据结构，确保类型安全和数据验证
"""

from datetime import date, datetime
from typing import List, Optional, Union, Dict, Any
from decimal import Decimal

from pydantic import BaseModel, Field, validator, root_validator


class StockDailyBar(BaseModel):
    """
    股票日线数据模型 - 简化版本，用于Parquet存储
    """
    date: date = Field(..., description="交易日期")
    open: float = Field(..., description="开盘价")
    close: float = Field(..., description="收盘价")
    high: float = Field(..., description="最高价")
    low: float = Field(..., description="最低价")
    volume: float = Field(..., description="成交量")
    amount: Optional[float] = Field(None, description="成交额")
    adjust_factor: Optional[float] = Field(None, description="复权因子")


class StockData(BaseModel):
    """
    单只股票的所有历史数据
    """
    symbol: str = Field(..., description="股票代码")
    name: Optional[str] = Field(None, description="股票名称")
    bars: List[StockDailyBar] = Field(..., description="日线数据列表")


class StockBasicInfo(BaseModel):
    """股票基本信息模型"""
    symbol: str = Field(..., description="股票代码，如000001")
    name: str = Field(..., description="股票名称")
    exchange: str = Field(..., description="交易所，SZ/SH")
    list_date: Optional[date] = Field(None, description="上市日期")
    is_st: bool = Field(False, description="是否ST股票")
    industry: Optional[str] = Field(None, description="所属行业")
    market_cap: Optional[float] = Field(None, description="总市值")
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }


class OHLCVRecord(BaseModel):
    """OHLCV数据记录模型 - 标准化的股票价格数据"""
    symbol: str = Field(..., description="股票代码")
    trade_date: date = Field(..., description="交易日期")
    open: Decimal = Field(..., ge=0, description="开盘价")
    high: Decimal = Field(..., ge=0, description="最高价")
    low: Decimal = Field(..., ge=0, description="最低价")
    close: Decimal = Field(..., ge=0, description="收盘价")
    volume: int = Field(..., ge=0, description="成交量(手)")
    amount: Optional[Decimal] = Field(None, ge=0, description="成交额(万元)")
    
    # 复权相关
    adj_close: Optional[Decimal] = Field(None, ge=0, description="复权收盘价")
    adj_factor: Optional[Decimal] = Field(None, description="复权因子")
    
    # 技术指标相关
    turnover_rate: Optional[Decimal] = Field(None, ge=0, description="换手率(%)")
    pe_ratio: Optional[Decimal] = Field(None, description="市盈率")
    pb_ratio: Optional[Decimal] = Field(None, description="市净率")
    
    @validator('high')
    def high_must_be_max(cls, v, values):
        """验证最高价必须是四个价格中的最高值"""
        if 'open' in values and 'low' in values and 'close' in values:
            prices = [values['open'], values['low'], values['close'], v]
            if v != max(prices):
                raise ValueError('最高价必须是开盘价、最高价、最低价、收盘价中的最高值')
        return v
    
    @validator('low')
    def low_must_be_min(cls, v, values):
        """验证最低价必须是四个价格中的最低值"""
        if 'open' in values and 'high' in values and 'close' in values:
            prices = [values['open'], values['high'], values['close'], v]
            if v != min(prices):
                raise ValueError('最低价必须是开盘价、最高价、最低价、收盘价中的最低值')
        return v
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat()
        }


class DataQualityReport(BaseModel):
    """数据质量报告模型"""
    symbol: str = Field(..., description="股票代码")
    total_records: int = Field(..., ge=0, description="总记录数")
    valid_records: int = Field(..., ge=0, description="有效记录数")
    missing_data_count: int = Field(..., ge=0, description="缺失数据数量")
    abnormal_price_count: int = Field(..., ge=0, description="异常价格数量")
    duplicate_date_count: int = Field(..., ge=0, description="重复日期数量")
    quality_score: float = Field(..., ge=0, le=1, description="质量评分(0-1)")
    start_date: date = Field(..., description="数据开始日期")
    end_date: date = Field(..., description="数据结束日期")
    issues: List[str] = Field(default_factory=list, description="发现的问题列表")
    
    @validator('valid_records')
    def valid_records_not_exceed_total(cls, v, values):
        """验证有效记录数不能超过总记录数"""
        if 'total_records' in values and v > values['total_records']:
            raise ValueError('有效记录数不能超过总记录数')
        return v
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }


class DataSyncRequest(BaseModel):
    """数据同步请求模型"""
    symbols: List[str] = Field(..., min_items=1, max_items=100, description="股票代码列表")
    start_date: date = Field(..., description="开始日期")
    end_date: date = Field(..., description="结束日期")
    adjust_type: str = Field("qfq", description="复权类型：qfq(前复权)/hfq(后复权)/none(不复权)")
    force_update: bool = Field(False, description="是否强制更新已存在的数据")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """验证股票代码格式"""
        for symbol in v:
            if not (symbol.isdigit() and len(symbol) == 6):
                raise ValueError(f'股票代码格式错误: {symbol}，应为6位数字')
        return v
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        """验证结束日期必须晚于开始日期"""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('结束日期必须晚于开始日期')
        return v
    
    @validator('adjust_type')
    def validate_adjust_type(cls, v):
        """验证复权类型"""
        valid_types = ['qfq', 'hfq', 'none']
        if v not in valid_types:
            raise ValueError(f'复权类型必须是{valid_types}中的一种')
        return v


class DataSyncResponse(BaseModel):
    """数据同步响应模型"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态：pending/running/completed/failed")
    symbols_requested: List[str] = Field(..., description="请求的股票代码")
    symbols_completed: List[str] = Field(default_factory=list, description="已完成的股票代码")
    symbols_failed: List[str] = Field(default_factory=list, description="失败的股票代码")
    total_records: int = Field(0, description="同步的总记录数")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    error_message: Optional[str] = Field(None, description="错误信息")
    quality_reports: List[DataQualityReport] = Field(default_factory=list, description="数据质量报告")


class AKShareApiError(BaseModel):
    """AKShare API错误信息模型"""
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误信息")
    symbol: Optional[str] = Field(None, description="相关股票代码")
    retry_count: int = Field(0, description="重试次数")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间")


class RetryConfig(BaseModel):
    """重试配置模型"""
    max_retries: int = Field(3, ge=0, le=10, description="最大重试次数")
    base_delay: float = Field(1.0, gt=0, description="基础延迟时间(秒)")
    max_delay: float = Field(60.0, gt=0, description="最大延迟时间(秒)")
    exponential_base: float = Field(2.0, gt=1, description="指数退避基数")
    jitter: bool = Field(True, description="是否添加随机抖动")


class DataSourceConfig(BaseModel):
    """数据源配置模型"""
    name: str = Field(..., description="数据源名称")
    enabled: bool = Field(True, description="是否启用")
    priority: int = Field(1, ge=1, description="优先级(1最高)")
    timeout: int = Field(30, gt=0, description="超时时间(秒)")
    retry_config: RetryConfig = Field(default_factory=RetryConfig, description="重试配置")
    rate_limit: Optional[int] = Field(None, description="频率限制(每分钟请求数)")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="额外参数")
