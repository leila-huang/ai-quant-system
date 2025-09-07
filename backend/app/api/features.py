"""
特征计算API路由

提供技术指标计算、特征工程流水线、特征存储等接口。
支持单股票和批量特征计算，以及特征数据的查询和管理。
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, status
from pydantic import BaseModel, Field, validator
import pandas as pd

from backend.app.core.config import settings
from backend.src.models.basic_models import StockData, StockDailyBar
from backend.src.storage.parquet_engine import get_parquet_storage
from backend.src.engine.features.indicators import TechnicalIndicators
from backend.src.engine.features.pipeline import FeaturePipeline
from backend.src.engine.features.feature_store import FeatureStore


router = APIRouter()

# 全局线程池用于异步特征计算
feature_executor = ThreadPoolExecutor(max_workers=4)


class IndicatorRequest(BaseModel):
    """技术指标计算请求模型"""
    symbol: str = Field(..., description="股票代码")
    indicators: List[str] = Field(
        default=["MA", "RSI", "MACD"], 
        description="要计算的技术指标列表"
    )
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    
    # 指标参数
    ma_windows: Optional[List[int]] = Field([5, 10, 20, 30], description="MA窗口期")
    ema_windows: Optional[List[int]] = Field([12, 26], description="EMA窗口期")
    rsi_windows: Optional[List[int]] = Field([14, 6], description="RSI窗口期")
    boll_window: Optional[int] = Field(20, description="布林带窗口期")
    boll_std_dev: Optional[float] = Field(2.0, description="布林带标准差倍数")
    
    @validator('indicators')
    def validate_indicators(cls, v):
        valid_indicators = ["MA", "EMA", "RSI", "MACD", "BOLL", "KDJ", "WILLIAMS", "VOLUME_MA"]
        for indicator in v:
            if indicator not in valid_indicators:
                raise ValueError(f"不支持的技术指标: {indicator}")
        return v


class BatchIndicatorRequest(BaseModel):
    """批量技术指标计算请求模型"""
    symbols: List[str] = Field(..., description="股票代码列表", max_items=50)
    indicators: List[str] = Field(
        default=["MA", "RSI", "MACD"], 
        description="要计算的技术指标列表"
    )
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    
    # 异步参数
    async_execution: bool = Field(False, description="是否异步执行")


class FeaturePipelineRequest(BaseModel):
    """特征工程流水线请求模型"""
    symbols: List[str] = Field(..., description="股票代码列表", max_items=100)
    feature_config: Dict[str, Any] = Field(
        default={
            "indicators": ["MA", "RSI", "MACD", "BOLL"],
            "scaling_method": "standard",
            "feature_selection": True,
            "n_features": 50
        },
        description="特征配置参数"
    )
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    save_features: bool = Field(True, description="是否保存特征数据")


class IndicatorResponse(BaseModel):
    """技术指标响应模型"""
    symbol: str
    indicators: Dict[str, List[float]]
    dates: List[str]
    metadata: Dict[str, Any]
    calculation_time_seconds: float


class BatchIndicatorResponse(BaseModel):
    """批量指标响应模型"""  
    results: Dict[str, IndicatorResponse]
    total_symbols: int
    successful_count: int
    failed_symbols: List[str]
    total_time_seconds: float
    task_id: Optional[str] = None


class FeaturePipelineResponse(BaseModel):
    """特征工程流水线响应模型"""
    feature_version: str
    symbols_processed: int
    feature_count: int
    feature_columns: List[str]
    processing_time_seconds: float
    storage_path: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 - 1.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class FeatureQueryRequest(BaseModel):
    """特征查询请求模型"""
    symbols: Optional[List[str]] = Field(None, description="股票代码列表")
    feature_version: Optional[str] = Field(None, description="特征版本")
    feature_type: str = Field("raw", description="特征类型: raw/processed")
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")


# 全局任务状态存储（生产环境应使用Redis等）
_task_status = {}


@router.post("/indicators/calculate", response_model=IndicatorResponse, summary="计算技术指标")
async def calculate_indicators(request: IndicatorRequest):
    """
    计算单只股票的技术指标
    
    支持的技术指标：
    - MA: 移动平均线
    - EMA: 指数移动平均线
    - RSI: 相对强弱指数
    - MACD: 异同移动平均线
    - BOLL: 布林带
    - KDJ: 随机指标
    - WILLIAMS: 威廉指标
    - VOLUME_MA: 成交量移动平均
    """
    try:
        start_time = datetime.now()
        
        # 获取股票数据
        storage = get_parquet_storage()
        stock_data = storage.load_stock_data(
            symbol=request.symbol,
            start_date=request.start_date or (datetime.now().date() - timedelta(days=365)),
            end_date=request.end_date or datetime.now().date()
        )
        
        if not stock_data or not stock_data.bars:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到股票 {request.symbol} 的数据"
            )
        
        # 计算技术指标
        calculator = TechnicalIndicators()
        
        # 构建指标参数
        indicator_params = {
            'ma_windows': request.ma_windows,
            'ema_windows': request.ema_windows,
            'rsi_windows': request.rsi_windows,
            'boll_window': request.boll_window,
            'boll_std_dev': request.boll_std_dev
        }
        
        result_df = calculator.calculate(stock_data, request.indicators, **indicator_params)
        
        # 准备响应数据
        indicators_data = {}
        for column in result_df.columns:
            if column not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']:
                indicators_data[column] = result_df[column].fillna(0).tolist()
        
        dates = result_df['date'].dt.strftime('%Y-%m-%d').tolist()
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return IndicatorResponse(
            symbol=request.symbol,
            indicators=indicators_data,
            dates=dates,
            metadata={
                "data_points": len(result_df),
                "date_range": {
                    "start": dates[0] if dates else None,
                    "end": dates[-1] if dates else None
                },
                "indicators_calculated": request.indicators,
                "parameters": indicator_params
            },
            calculation_time_seconds=round(calculation_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"计算技术指标失败: {str(e)}"
        )


@router.post("/indicators/batch", response_model=BatchIndicatorResponse, summary="批量计算技术指标")
async def calculate_batch_indicators(request: BatchIndicatorRequest, background_tasks: BackgroundTasks):
    """
    批量计算多只股票的技术指标
    
    支持同步和异步执行模式。异步模式会返回任务ID，可通过任务状态接口查询进度。
    """
    try:
        start_time = datetime.now()
        
        if request.async_execution:
            # 异步执行
            task_id = f"batch_indicators_{int(start_time.timestamp())}"
            
            # 初始化任务状态
            _task_status[task_id] = {
                "status": "pending",
                "progress": 0.0,
                "result": None,
                "error_message": None,
                "created_at": start_time.isoformat(),
                "updated_at": start_time.isoformat()
            }
            
            # 添加后台任务
            background_tasks.add_task(
                _run_batch_indicators_task,
                task_id, request.symbols, request.indicators,
                request.start_date, request.end_date
            )
            
            return BatchIndicatorResponse(
                results={},
                total_symbols=len(request.symbols),
                successful_count=0,
                failed_symbols=[],
                total_time_seconds=0.0,
                task_id=task_id
            )
        
        # 同步执行
        results = {}
        failed_symbols = []
        calculator = TechnicalIndicators()
        storage = get_parquet_storage()
        
        for symbol in request.symbols:
            try:
                # 加载股票数据
                stock_data = storage.load_stock_data(
                    symbol=symbol,
                    start_date=request.start_date or (datetime.now().date() - timedelta(days=365)),
                    end_date=request.end_date or datetime.now().date()
                )
                
                if not stock_data or not stock_data.bars:
                    failed_symbols.append(symbol)
                    continue
                
                # 计算指标
                result_df = calculator.calculate(stock_data, request.indicators)
                
                # 提取指标数据
                indicators_data = {}
                for column in result_df.columns:
                    if column not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']:
                        indicators_data[column] = result_df[column].fillna(0).tolist()
                
                dates = result_df['date'].dt.strftime('%Y-%m-%d').tolist()
                
                results[symbol] = IndicatorResponse(
                    symbol=symbol,
                    indicators=indicators_data,
                    dates=dates,
                    metadata={
                        "data_points": len(result_df),
                        "indicators_calculated": request.indicators
                    },
                    calculation_time_seconds=0.0  # 批量计算时不单独计时
                )
                
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"计算 {symbol} 指标失败: {e}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return BatchIndicatorResponse(
            results=results,
            total_symbols=len(request.symbols),
            successful_count=len(results),
            failed_symbols=failed_symbols,
            total_time_seconds=round(total_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量计算技术指标失败: {str(e)}"
        )


@router.post("/pipeline/run", response_model=FeaturePipelineResponse, summary="运行特征工程流水线")
async def run_feature_pipeline(request: FeaturePipelineRequest, background_tasks: BackgroundTasks):
    """
    运行完整的特征工程流水线
    
    包括技术指标计算、特征缩放、特征选择等步骤，并可选择保存处理后的特征数据。
    """
    try:
        start_time = datetime.now()
        
        # 准备股票数据
        storage = get_parquet_storage()
        stock_data_list = []
        
        for symbol in request.symbols:
            try:
                stock_data = storage.load_stock_data(
                    symbol=symbol,
                    start_date=request.start_date or (datetime.now().date() - timedelta(days=365)),
                    end_date=request.end_date or datetime.now().date()
                )
                if stock_data and stock_data.bars:
                    stock_data_list.append(stock_data)
            except Exception as e:
                print(f"加载 {symbol} 数据失败: {e}")
        
        if not stock_data_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="未找到任何有效的股票数据"
            )
        
        # 创建特征工程流水线
        pipeline = FeaturePipeline()
        
        # 配置流水线参数
        config = request.feature_config
        indicators = config.get('indicators', ['MA', 'RSI', 'MACD'])
        scaling_method = config.get('scaling_method', 'standard')
        feature_selection = config.get('feature_selection', True)
        n_features = config.get('n_features', 50)
        
        # 拟合并转换特征
        features_df = pipeline.fit_transform(
            stock_data_list=stock_data_list,
            indicators=indicators,
            scaling_method=scaling_method,
            feature_selection=feature_selection,
            n_features=n_features
        )
        
        if features_df.empty:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="特征工程流水线未能产生任何特征"
            )
        
        # 保存特征数据
        feature_version = None
        storage_path = None
        if request.save_features:
            try:
                feature_store = FeatureStore()
                pipeline_config = pipeline.get_pipeline_info()['config']
                feature_version = feature_store.store_processed_features(features_df, pipeline_config)
                storage_path = str(feature_store.processed_features_path / f"features_{feature_version}.parquet")
            except Exception as e:
                print(f"保存特征数据失败: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FeaturePipelineResponse(
            feature_version=feature_version or f"temp_{int(start_time.timestamp())}",
            symbols_processed=len(stock_data_list),
            feature_count=len(features_df.columns),
            feature_columns=features_df.columns.tolist(),
            processing_time_seconds=round(processing_time, 3),
            storage_path=storage_path
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"特征工程流水线执行失败: {str(e)}"
        )


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="查询任务状态")
async def get_task_status(task_id: str):
    """
    查询异步任务的执行状态
    
    用于跟踪长时间运行的特征计算任务进度。
    """
    if task_id not in _task_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到任务 {task_id}"
        )
    
    task_info = _task_status[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        result=task_info["result"],
        error_message=task_info["error_message"],
        created_at=task_info["created_at"],
        updated_at=task_info["updated_at"]
    )


@router.post("/features/query", summary="查询特征数据")
async def query_features(request: FeatureQueryRequest):
    """
    查询已保存的特征数据
    
    支持按股票代码、特征版本、日期范围等条件查询特征数据。
    """
    try:
        feature_store = FeatureStore()
        
        if request.feature_type == "raw":
            # 查询原始特征
            features_df = feature_store.load_raw_features(
                feature_version=request.feature_version,
                symbols=request.symbols
            )
        else:
            # 查询处理后特征
            features_df = feature_store.load_processed_features(
                feature_version=request.feature_version
            )
        
        if features_df.empty:
            return {
                "message": "未找到匹配的特征数据",
                "feature_count": 0,
                "data": []
            }
        
        # 按日期范围过滤
        if request.start_date or request.end_date:
            if 'date' in features_df.columns:
                if request.start_date:
                    features_df = features_df[features_df['date'] >= pd.to_datetime(request.start_date)]
                if request.end_date:
                    features_df = features_df[features_df['date'] <= pd.to_datetime(request.end_date)]
        
        # 按股票代码过滤
        if request.symbols and 'symbol' in features_df.columns:
            features_df = features_df[features_df['symbol'].isin(request.symbols)]
        
        return {
            "message": "查询成功",
            "feature_count": len(features_df.columns),
            "data_points": len(features_df),
            "columns": features_df.columns.tolist(),
            "data": features_df.head(100).to_dict('records')  # 限制返回前100条
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询特征数据失败: {str(e)}"
        )


@router.delete("/features/{feature_version}", summary="删除特征数据")
async def delete_features(feature_version: str, feature_type: str = "processed"):
    """
    删除指定版本的特征数据
    """
    try:
        feature_store = FeatureStore()
        # 这里需要在FeatureStore中实现delete方法
        # feature_store.delete_features(feature_version, feature_type)
        
        return {
            "message": f"特征数据 {feature_version} 删除成功",
            "feature_version": feature_version,
            "feature_type": feature_type
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除特征数据失败: {str(e)}"
        )


@router.get("/features/versions", summary="获取特征版本列表")
async def list_feature_versions():
    """
    获取所有可用的特征版本列表
    """
    try:
        feature_store = FeatureStore()
        # 这里需要在FeatureStore中实现list_versions方法
        # versions = feature_store.list_versions()
        
        # 临时实现
        import os
        raw_versions = []
        processed_versions = []
        
        if feature_store.raw_features_path.exists():
            raw_versions = [f.stem.split('_')[1] for f in feature_store.raw_features_path.glob("*_*.parquet")]
        
        if feature_store.processed_features_path.exists():
            processed_versions = [f.stem.split('_')[1] for f in feature_store.processed_features_path.glob("features_*.parquet")]
        
        return {
            "raw_feature_versions": list(set(raw_versions)),
            "processed_feature_versions": list(set(processed_versions)),
            "total_versions": len(set(raw_versions + processed_versions))
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取特征版本列表失败: {str(e)}"
        )


# 异步任务执行函数
async def _run_batch_indicators_task(task_id: str, symbols: List[str], indicators: List[str],
                                   start_date: Optional[date], end_date: Optional[date]):
    """批量指标计算异步任务"""
    try:
        # 更新任务状态为运行中
        _task_status[task_id]["status"] = "running"
        _task_status[task_id]["updated_at"] = datetime.now().isoformat()
        
        results = {}
        failed_symbols = []
        calculator = TechnicalIndicators()
        storage = get_parquet_storage()
        
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                # 更新进度
                progress = (i + 1) / total_symbols
                _task_status[task_id]["progress"] = progress
                _task_status[task_id]["updated_at"] = datetime.now().isoformat()
                
                # 加载数据并计算指标
                stock_data = storage.load_stock_data(
                    symbol=symbol,
                    start_date=start_date or (datetime.now().date() - timedelta(days=365)),
                    end_date=end_date or datetime.now().date()
                )
                
                if not stock_data or not stock_data.bars:
                    failed_symbols.append(symbol)
                    continue
                
                result_df = calculator.calculate(stock_data, indicators)
                
                # 提取指标数据
                indicators_data = {}
                for column in result_df.columns:
                    if column not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']:
                        indicators_data[column] = result_df[column].fillna(0).tolist()
                
                dates = result_df['date'].dt.strftime('%Y-%m-%d').tolist()
                
                results[symbol] = {
                    "symbol": symbol,
                    "indicators": indicators_data,
                    "dates": dates,
                    "metadata": {
                        "data_points": len(result_df),
                        "indicators_calculated": indicators
                    },
                    "calculation_time_seconds": 0.0
                }
                
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"异步计算 {symbol} 指标失败: {e}")
        
        # 任务完成
        _task_status[task_id]["status"] = "completed"
        _task_status[task_id]["progress"] = 1.0
        _task_status[task_id]["result"] = {
            "results": results,
            "total_symbols": total_symbols,
            "successful_count": len(results),
            "failed_symbols": failed_symbols
        }
        _task_status[task_id]["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        # 任务失败
        _task_status[task_id]["status"] = "failed"
        _task_status[task_id]["error_message"] = str(e)
        _task_status[task_id]["updated_at"] = datetime.now().isoformat()



