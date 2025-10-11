"""
回测执行API路由

提供策略回测、回测结果查询、报告生成等接口。
支持vectorbt高性能回测引擎，包含A股约束、交易成本等真实市场模拟。
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import logging

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, status, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

from backend.app.core.config import settings
from backend.src.models.basic_models import StockData
from backend.src.engine.backtest.vectorbt_engine import TradingConfig
from backend.src.engine.backtest.constraints import MarketType, StockStatus
from backend.src.storage.parquet_engine import get_parquet_storage
from backend.src.engine.backtest.vectorbt_engine import VectorbtBacktestEngine
from backend.src.engine.backtest.constraints import AStockConstraints, ConstraintConfig, StockInfo
from backend.src.engine.backtest.cost_model import TradingCostModel, CostConfig, BrokerType
from backend.src.engine.backtest.report_generator import BacktestReportGenerator, ReportConfig
from backend.src.engine.backtest.metrics import PerformanceAnalyzer
from backend.app.api.websocket import broadcast_backtest_progress


router = APIRouter()

# 模块级日志记录器
logger = logging.getLogger(__name__)

# 全局线程池用于异步回测
backtest_executor = ThreadPoolExecutor(max_workers=3)

# 回测结果存储路径
BACKTEST_STORAGE_PATH = Path("backtests")
BACKTEST_STORAGE_PATH.mkdir(exist_ok=True)

# 全局任务状态存储
_backtest_tasks = {}


class StrategyConfig(BaseModel):
    """策略配置模型"""
    strategy_type: str = Field(..., description="策略类型")
    strategy_name: str = Field(..., description="策略名称")
    parameters: Dict[str, Any] = Field(default={}, description="策略参数")
    
    @validator('strategy_type')
    def validate_strategy_type(cls, v):
        valid_types = ["ma_crossover", "rsi_mean_reversion", "momentum", "custom"]
        if v not in valid_types:
            raise ValueError(f"不支持的策略类型: {v}")
        return v


class BacktestRequest(BaseModel):
    """回测请求模型"""
    backtest_name: str = Field(..., description="回测名称")
    strategy_config: StrategyConfig = Field(..., description="策略配置")
    universe: List[str] = Field(..., description="股票池", max_items=500)
    start_date: date = Field(..., description="回测开始日期")
    end_date: date = Field(..., description="回测结束日期")
    
    # 交易配置
    initial_capital: float = Field(1000000.0, description="初始资金")
    commission: float = Field(0.0003, description="手续费率")
    enable_cost_model: bool = Field(True, description="是否启用交易成本模型")
    broker_type: str = Field("standard", description="券商类型")
    
    # A股约束配置
    enable_constraints: bool = Field(True, description="是否启用A股约束")
    price_limit_enabled: bool = Field(True, description="是否启用涨跌停约束")
    t_plus_1_enabled: bool = Field(True, description="是否启用T+1约束")
    
    # 回测配置
    rebalance_frequency: str = Field("daily", description="调仓频率")
    benchmark_symbol: Optional[str] = Field("000300.SH", description="基准指数")
    
    # 异步执行
    async_execution: bool = Field(False, description="是否异步执行")
    
    @validator('broker_type')
    def validate_broker_type(cls, v):
        valid_types = ["standard", "low_commission", "minimum"]
        if v not in valid_types:
            raise ValueError(f"不支持的券商类型: {v}")
        return v
    
    @validator('rebalance_frequency')
    def validate_rebalance_frequency(cls, v):
        valid_frequencies = ["daily", "weekly", "monthly"]
        if v not in valid_frequencies:
            raise ValueError(f"不支持的调仓频率: {v}")
        return v


class BacktestResponse(BaseModel):
    """回测响应模型"""
    backtest_id: str
    backtest_name: str
    status: str
    performance_metrics: Dict[str, float]
    execution_time_seconds: float
    trade_count: int
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    task_id: Optional[str] = None


class BacktestStatusResponse(BaseModel):
    """回测状态响应模型"""
    backtest_id: str
    status: str
    progress: float
    current_step: str
    estimated_time_remaining: Optional[float] = None
    error_message: Optional[str] = None
    started_at: str
    updated_at: str


class BacktestListResponse(BaseModel):
    """回测列表响应模型"""
    backtests: List[Dict[str, Any]]
    total_count: int


class ReportRequest(BaseModel):
    """报告生成请求模型"""
    backtest_id: str = Field(..., description="回测ID")
    report_format: str = Field("html", description="报告格式: html/pdf/json/excel")
    include_charts: bool = Field(True, description="是否包含图表")
    include_trades: bool = Field(True, description="是否包含交易明细")
    custom_title: Optional[str] = Field(None, description="自定义报告标题")


class ComparisonRequest(BaseModel):
    """回测对比请求模型"""
    backtest_ids: List[str] = Field(..., description="要对比的回测ID列表", min_items=2, max_items=10)
    comparison_metrics: List[str] = Field(
        default=["total_return", "annual_return", "sharpe_ratio", "max_drawdown"],
        description="对比指标"
    )
    report_format: str = Field("html", description="对比报告格式")


@router.post("/run", response_model=BacktestResponse, summary="运行回测")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    运行策略回测
    
    支持vectorbt高性能回测引擎，包含完整的A股约束和交易成本模拟。
    可选择同步或异步执行模式。
    """
    try:
        start_time = datetime.now()
        backtest_id = f"{request.backtest_name}_{uuid.uuid4().hex[:8]}"
        
        if request.async_execution:
            # 异步执行
            task_id = f"backtest_{backtest_id}_{int(start_time.timestamp())}"
            
            # 初始化任务状态
            _backtest_tasks[task_id] = {
                "status": "pending",
                "progress": 0.0,
                "current_step": "初始化",
                "result": None,
                "error_message": None,
                "started_at": start_time.isoformat(),
                "updated_at": start_time.isoformat()
            }
            
            # 添加后台任务
            background_tasks.add_task(
                _run_backtest_task,
                task_id, backtest_id, request
            )
            
            return BacktestResponse(
                backtest_id=backtest_id,
                backtest_name=request.backtest_name,
                status="pending",
                performance_metrics={},
                execution_time_seconds=0.0,
                trade_count=0,
                total_return=0.0,
                annual_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                task_id=task_id
            )
        
        # 同步执行
        result = await _execute_backtest(backtest_id, request)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        result.execution_time_seconds = round(execution_time, 3)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"回测执行失败: {str(e)}"
        )


@router.get("/{backtest_id}/status", response_model=BacktestStatusResponse, summary="查询回测状态")
async def get_backtest_status(backtest_id: str):
    """
    查询回测执行状态
    
    用于跟踪异步回测任务的进度，包括当前步骤和预估剩余时间。
    """
    try:
        # 查找对应的任务ID
        task_id = None
        for tid, task_info in _backtest_tasks.items():
            if backtest_id in tid:
                task_id = tid
                break
        
        if not task_id:
            # 检查是否已完成的回测
            backtest_file = BACKTEST_STORAGE_PATH / f"{backtest_id}_metadata.json"
            if backtest_file.exists():
                with open(backtest_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                return BacktestStatusResponse(
                    backtest_id=backtest_id,
                    status="completed",
                    progress=1.0,
                    current_step="已完成",
                    started_at=metadata.get("started_at", ""),
                    updated_at=metadata.get("completed_at", "")
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"未找到回测 {backtest_id}"
                )
        
        task_info = _backtest_tasks[task_id]
        
        return BacktestStatusResponse(
            backtest_id=backtest_id,
            status=task_info["status"],
            progress=task_info["progress"],
            current_step=task_info.get("current_step", "未知"),
            error_message=task_info.get("error_message"),
            started_at=task_info["started_at"],
            updated_at=task_info["updated_at"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询回测状态失败: {str(e)}"
        )


@router.get("/list", response_model=BacktestListResponse, summary="获取回测列表")
async def list_backtests(
    status_filter: Optional[str] = Query(None, description="状态筛选"),
    start_date: Optional[date] = Query(None, description="开始日期筛选"),
    limit: int = Query(50, description="返回数量限制")
):
    """
    获取回测历史列表
    
    支持按状态、日期等条件筛选，返回回测的基本信息和性能摘要。
    """
    try:
        backtests = []
        
        # 扫描回测存储目录
        for metadata_file in BACKTEST_STORAGE_PATH.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 应用筛选条件
                if status_filter and metadata.get("status") != status_filter:
                    continue
                
                if start_date:
                    backtest_date = datetime.fromisoformat(metadata.get("started_at", "")).date()
                    if backtest_date < start_date:
                        continue
                
                backtests.append(metadata)
                
                if len(backtests) >= limit:
                    break
                    
            except Exception as e:
                print(f"读取回测元数据失败 {metadata_file}: {e}")
                continue
        
        # 按开始时间排序
        backtests.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        
        return BacktestListResponse(
            backtests=backtests,
            total_count=len(backtests)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取回测列表失败: {str(e)}"
        )


@router.get("/{backtest_id}/results", summary="获取回测详细结果")
async def get_backtest_results(backtest_id: str):
    """
    获取回测的详细结果数据
    
    包括完整的业绩指标、持仓记录、交易明细等。
    """
    try:
        # 检查回测是否存在
        metadata_file = BACKTEST_STORAGE_PATH / f"{backtest_id}_metadata.json"
        results_file = BACKTEST_STORAGE_PATH / f"{backtest_id}_results.json"
        
        if not metadata_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到回测 {backtest_id}"
            )
        
        # 读取元数据
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 读取详细结果
        results_data = {}
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
        
        return {
            "backtest_id": backtest_id,
            "metadata": metadata,
            "results": results_data,
            "status": metadata.get("status", "unknown")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取回测结果失败: {str(e)}"
        )


@router.post("/{backtest_id}/report", summary="生成回测报告")
async def generate_backtest_report(backtest_id: str, request: ReportRequest):
    """
    生成回测报告
    
    支持HTML、PDF、JSON、Excel等多种格式的专业回测报告。
    """
    try:
        # 检查回测是否存在
        results_file = BACKTEST_STORAGE_PATH / f"{backtest_id}_results.json"
        if not results_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到回测结果 {backtest_id}"
            )
        
        # 读取回测结果
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # 重构BacktestResult对象 (简化实现)
        returns = pd.Series(results_data.get("returns", {}))
        returns.index = pd.to_datetime(returns.index)
        
        trades_data = results_data.get("trades", [])
        trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
        
        positions_data = results_data.get("positions", [])
        positions_df = pd.DataFrame(positions_data) if positions_data else pd.DataFrame()
        
        # 创建BacktestResult对象
        from backend.src.engine.backtest import BacktestResult
        backtest_result = BacktestResult(
            returns=returns,
            positions=positions_df,
            trades=trades_df,
            metrics=results_data.get("metrics", {}),
            metadata=results_data.get("metadata", {})
        )
        
        # 配置报告生成器
        report_config = ReportConfig(
            report_title=request.custom_title or f"回测报告 - {backtest_id}",
            include_charts=request.include_charts,
            include_trade_analysis=request.include_trades
        )
        
        # 生成报告
        report_generator = BacktestReportGenerator(report_config)
        report_path = report_generator.generate_report(backtest_result, request.report_format)
        
        # 返回文件下载响应
        if request.report_format.lower() in ["html", "pdf", "json"]:
            return FileResponse(
                path=report_path,
                filename=f"backtest_report_{backtest_id}.{request.report_format}",
                media_type='application/octet-stream'
            )
        else:
            return {
                "message": "报告生成成功",
                "report_path": report_path,
                "report_format": request.report_format
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成回测报告失败: {str(e)}"
        )


@router.post("/compare", summary="对比多个回测结果")
async def compare_backtests(request: ComparisonRequest):
    """
    对比多个回测结果
    
    生成策略对比分析报告，包括性能指标对比、收益曲线对比等。
    """
    try:
        strategies_returns = {}
        comparison_data = {}
        
        # 收集所有回测的数据
        for backtest_id in request.backtest_ids:
            results_file = BACKTEST_STORAGE_PATH / f"{backtest_id}_results.json"
            metadata_file = BACKTEST_STORAGE_PATH / f"{backtest_id}_metadata.json"
            
            if not results_file.exists() or not metadata_file.exists():
                continue
            
            # 读取数据
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 构建收益率序列
            returns = pd.Series(results_data.get("returns", {}))
            returns.index = pd.to_datetime(returns.index)
            
            strategy_name = metadata.get("backtest_name", backtest_id)
            strategies_returns[strategy_name] = returns
            
            # 收集对比指标
            metrics = results_data.get("metrics", {})
            comparison_data[strategy_name] = {
                "backtest_id": backtest_id,
                "strategy_config": metadata.get("strategy_config", {}),
                "performance_metrics": {
                    metric: metrics.get(metric, 0.0) 
                    for metric in request.comparison_metrics
                }
            }
        
        if not strategies_returns:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="未找到有效的回测结果进行对比"
            )
        
        # 生成对比报告
        analyzer = PerformanceAnalyzer()
        comparison_df = analyzer.compare_strategies(strategies_returns)
        
        if request.report_format == "html":
            # 这里可以生成HTML对比报告
            return {
                "message": "策略对比分析完成",
                "comparison_data": comparison_data,
                "comparison_summary": comparison_df.to_dict(),
                "strategies_count": len(strategies_returns)
            }
        else:
            return {
                "comparison_data": comparison_data,
                "comparison_metrics": comparison_df.to_dict(),
                "report_format": request.report_format
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"回测对比失败: {str(e)}"
        )


@router.delete("/{backtest_id}", summary="删除回测")
async def delete_backtest(backtest_id: str):
    """
    删除指定的回测结果和相关文件
    """
    try:
        deleted_files = []
        
        # 删除元数据文件
        metadata_file = BACKTEST_STORAGE_PATH / f"{backtest_id}_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
            deleted_files.append("metadata")
        
        # 删除结果文件
        results_file = BACKTEST_STORAGE_PATH / f"{backtest_id}_results.json"
        if results_file.exists():
            results_file.unlink()
            deleted_files.append("results")
        
        if not deleted_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到回测 {backtest_id}"
            )
        
        return {
            "message": f"回测 {backtest_id} 删除成功",
            "deleted_files": deleted_files
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除回测失败: {str(e)}"
        )


@router.get("/strategies/supported", summary="获取支持的策略类型")
async def get_supported_strategies():
    """
    获取系统支持的所有策略类型和参数说明
    """
    strategies = {
        "ma_crossover": {
            "name": "移动平均线穿越",
            "description": "基于快慢均线金叉死叉的趋势跟踪策略",
            "parameters": {
                "fast_period": {"type": "int", "default": 5, "description": "快线周期"},
                "slow_period": {"type": "int", "default": 20, "description": "慢线周期"},
                "stop_loss": {"type": "float", "default": 0.05, "description": "止损比例"}
            }
        },
        "rsi_mean_reversion": {
            "name": "RSI均值回归",
            "description": "基于RSI超买超卖的均值回归策略",
            "parameters": {
                "rsi_period": {"type": "int", "default": 14, "description": "RSI周期"},
                "oversold": {"type": "int", "default": 30, "description": "超卖阈值"},
                "overbought": {"type": "int", "default": 70, "description": "超买阈值"}
            }
        },
        "momentum": {
            "name": "动量策略",
            "description": "基于价格动量的趋势策略",
            "parameters": {
                "lookback": {"type": "int", "default": 20, "description": "回望周期"},
                "threshold": {"type": "float", "default": 0.02, "description": "动量阈值"}
            }
        }
    }
    
    return {
        "supported_strategies": strategies,
        "total_count": len(strategies)
    }


# 异步任务执行函数
async def _run_backtest_task(task_id: str, backtest_id: str, request: BacktestRequest):
    """回测异步任务"""
    try:
        # 更新任务状态为运行中
        _backtest_tasks[task_id]["status"] = "running"
        _backtest_tasks[task_id]["current_step"] = "开始执行回测"
        _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 执行回测
        result = await _execute_backtest(backtest_id, request, task_id)
        
        # 任务完成
        _backtest_tasks[task_id]["status"] = "completed"
        _backtest_tasks[task_id]["progress"] = 1.0
        _backtest_tasks[task_id]["current_step"] = "回测完成"
        _backtest_tasks[task_id]["result"] = result.dict()
        _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        # 任务失败
        _backtest_tasks[task_id]["status"] = "failed"
        _backtest_tasks[task_id]["current_step"] = "回测失败"
        _backtest_tasks[task_id]["error_message"] = str(e)
        _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()


async def _execute_backtest(backtest_id: str, request: BacktestRequest, task_id: str = None) -> BacktestResponse:
    """执行回测的核心逻辑"""
    try:
        # 更新进度：初始化回测引擎
        if task_id:
            _backtest_tasks[task_id]["progress"] = 0.1
            _backtest_tasks[task_id]["current_step"] = "初始化回测引擎"
            _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # WebSocket推送进度
            await broadcast_backtest_progress(
                backtest_id=backtest_id,
                status="running", 
                progress=0.1,
                current_step="初始化回测引擎"
            )
        
        # 创建交易配置
        trading_config = TradingConfig(
            initial_cash=request.initial_capital,
            commission=request.commission
        )
        
        # 创建A股约束
        constraints = None
        if request.enable_constraints:
            constraint_config = ConstraintConfig(
                enable_price_limit=request.price_limit_enabled,
                enable_t_plus_1=request.t_plus_1_enabled
            )
            constraints = AStockConstraints(constraint_config)
        
        # 创建成本模型
        cost_model = None
        if request.enable_cost_model:
            broker_type_map = {
                "standard": BrokerType.STANDARD,
                "low_commission": BrokerType.DISCOUNT,
                "minimum": BrokerType.DISCOUNT
            }
            cost_config = CostConfig(broker_type=broker_type_map.get(request.broker_type, BrokerType.STANDARD))
            cost_model = TradingCostModel(config=cost_config)
        
        # 创建回测引擎
        engine = VectorbtBacktestEngine(
            config=trading_config,
            constraints=constraints,
            enable_constraints=request.enable_constraints,
            cost_model=cost_model,
            enable_cost_model=request.enable_cost_model
        )
        
        # 更新进度：加载数据
        if task_id:
            _backtest_tasks[task_id]["progress"] = 0.3
            _backtest_tasks[task_id]["current_step"] = "加载股票数据"
            _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # WebSocket推送进度
            await broadcast_backtest_progress(
                backtest_id=backtest_id,
                status="running",
                progress=0.3,
                current_step="加载股票数据"
            )
        
        # 数据预加载和验证逻辑
        from backend.src.storage.parquet_engine import get_parquet_storage
        from backend.src.engine.features.indicators import TechnicalIndicators
        
        storage = get_parquet_storage()
        indicator_calculator = TechnicalIndicators()
        
        # 预加载和验证所有股票数据
        validated_data = await _preload_and_validate_data(
            storage, request.universe[:10], request.start_date, request.end_date, task_id, backtest_id
        )
        
        if not validated_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="没有找到有效的股票数据，请检查股票代码和日期范围"
            )
        
        # 使用验证后的数据生成交易信号
        signals_data = {}
        strategy_type = request.strategy_config.strategy_type
        strategy_params = request.strategy_config.parameters
        
        # 更新进度：生成交易信号
        if task_id:
            _backtest_tasks[task_id]["progress"] = 0.5
            _backtest_tasks[task_id]["current_step"] = "生成交易信号"
            _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # WebSocket推送进度
            await broadcast_backtest_progress(
                backtest_id=backtest_id,
                status="running",
                progress=0.5,
                current_step="生成交易信号"
            )
        
        for symbol, df in validated_data.items():
            try:
                # 根据策略类型生成信号
                if strategy_type == "ma_crossover":
                    signals = _generate_ma_crossover_signals(df, indicator_calculator, strategy_params)
                elif strategy_type == "rsi_mean_reversion":
                    signals = _generate_rsi_signals(df, indicator_calculator, strategy_params)
                elif strategy_type == "momentum":
                    signals = _generate_momentum_signals(df, indicator_calculator, strategy_params)
                else:
                    # 默认使用双均线策略
                    signals = _generate_ma_crossover_signals(df, indicator_calculator, strategy_params)
                
                signals_data[symbol] = signals
                
            except Exception as e:
                logger.error(f"生成股票 {symbol} 交易信号时出错: {e}")
                continue
        
        signals_df = pd.DataFrame(signals_data) if signals_data else pd.DataFrame()
        
        # 更新进度：执行回测
        if task_id:
            _backtest_tasks[task_id]["progress"] = 0.6
            _backtest_tasks[task_id]["current_step"] = "执行策略回测"
            _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # WebSocket推送进度
            await broadcast_backtest_progress(
                backtest_id=backtest_id,
                status="running",
                progress=0.6,
                current_step="执行策略回测"
            )
        
        # 运行回测
        backtest_result = engine.run_backtest(
            symbols=request.universe,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy_names=request.strategy_config.strategy_type
        )
        
        # 更新进度：计算性能指标
        if task_id:
            _backtest_tasks[task_id]["progress"] = 0.8
            _backtest_tasks[task_id]["current_step"] = "计算性能指标"
            _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # WebSocket推送进度
            await broadcast_backtest_progress(
                backtest_id=backtest_id,
                status="running",
                progress=0.8,
                current_step="计算性能指标"
            )
        
        # 提取关键指标
        metrics = backtest_result.metrics
        total_return = metrics.get('total_return', 0.0)
        annual_return = metrics.get('annualized_return', 0.0)
        max_drawdown = metrics.get('max_drawdown', 0.0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        
        # 保存回测结果
        metadata = {
            "backtest_id": backtest_id,
            "backtest_name": request.backtest_name,
            "strategy_config": request.strategy_config.dict(),
            "universe": request.universe,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "status": "completed",
            "performance_metrics": metrics,
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat()
        }
        
        # 保存元数据
        metadata_path = BACKTEST_STORAGE_PATH / f"{backtest_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果
        results_data = backtest_result.to_dict()
        results_path = BACKTEST_STORAGE_PATH / f"{backtest_id}_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)
        
        # WebSocket推送完成状态
        if task_id:
            await broadcast_backtest_progress(
                backtest_id=backtest_id,
                status="completed",
                progress=1.0,
                current_step="回测完成"
            )
        
        return BacktestResponse(
            backtest_id=backtest_id,
            backtest_name=request.backtest_name,
            status="completed",
            performance_metrics=metrics,
            execution_time_seconds=0.0,  # 会在外层设置
            trade_count=len(backtest_result.trades) if not backtest_result.trades.empty else 0,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )
        
    except Exception as e:
        # WebSocket推送错误状态
        if task_id:
            await broadcast_backtest_progress(
                backtest_id=backtest_id,
                status="failed",
                progress=0.0,
                current_step="回测失败",
                error_message=str(e)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"回测执行失败: {str(e)}"
        )


def _generate_ma_crossover_signals(df: pd.DataFrame, indicator_calculator, params: dict) -> pd.Series:
    """生成双均线交叉策略信号"""
    fast_period = params.get('fast_period', 5)
    slow_period = params.get('slow_period', 20)
    
    # 计算移动平均线
    df_with_ma = indicator_calculator._add_ma_indicators(df, ma_windows=[fast_period, slow_period])
    
    # 生成交叉信号
    fast_ma = df_with_ma[f'ma_{fast_period}']
    slow_ma = df_with_ma[f'ma_{slow_period}']
    
    # 信号：1=买入，-1=卖出，0=持有
    signals = pd.Series(0, index=df_with_ma.index)
    
    # 快线上穿慢线：买入信号
    golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    signals[golden_cross] = 1
    
    # 快线下穿慢线：卖出信号
    death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    signals[death_cross] = -1
    
    return signals


def _generate_rsi_signals(df: pd.DataFrame, indicator_calculator, params: dict) -> pd.Series:
    """生成RSI均值回归策略信号"""
    rsi_period = params.get('rsi_period', 14)
    oversold_threshold = params.get('oversold_threshold', 30)
    overbought_threshold = params.get('overbought_threshold', 70)
    
    # 计算RSI
    df_with_rsi = indicator_calculator._add_rsi_indicators(df, rsi_windows=[rsi_period])
    rsi = df_with_rsi[f'rsi_{rsi_period}']
    
    # 生成信号
    signals = pd.Series(0, index=df_with_rsi.index)
    
    # RSI < 30：超卖，买入信号
    oversold = rsi < oversold_threshold
    signals[oversold] = 1
    
    # RSI > 70：超买，卖出信号
    overbought = rsi > overbought_threshold
    signals[overbought] = -1
    
    return signals


def _generate_momentum_signals(df: pd.DataFrame, indicator_calculator, params: dict) -> pd.Series:
    """生成动量策略信号"""
    lookback_period = params.get('lookback_period', 20)
    momentum_threshold = params.get('momentum_threshold', 0.02)  # 2%
    
    # 计算价格动量
    close_prices = df['close']
    momentum = (close_prices / close_prices.shift(lookback_period) - 1)
    
    # 生成信号
    signals = pd.Series(0, index=df.index)
    
    # 正动量超过阈值：买入信号
    strong_momentum = momentum > momentum_threshold
    signals[strong_momentum] = 1
    
    # 负动量超过阈值：卖出信号
    weak_momentum = momentum < -momentum_threshold
    signals[weak_momentum] = -1
    
    return signals


# 数据缓存 - 使用模块级变量避免重复加载
_data_cache = {}
_cache_max_size = 100  # 最大缓存数量


async def _preload_and_validate_data(storage, symbols: List[str], start_date: date, end_date: date, 
                                   task_id: str = None, backtest_id: str = None) -> Dict[str, pd.DataFrame]:
    """
    预加载和验证股票数据
    
    Args:
        storage: 数据存储引擎
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        task_id: 任务ID（可选）
        backtest_id: 回测ID（可选）
        
    Returns:
        Dict[str, pd.DataFrame]: 验证后的股票数据字典
    """
    validated_data = {}
    missing_symbols = []
    invalid_symbols = []
    
    logger.info(f"开始预加载 {len(symbols)} 只股票的数据...")
    
    for i, symbol in enumerate(symbols):
        try:
            # 生成缓存键
            cache_key = f"{symbol}_{start_date}_{end_date}"
            
            # 检查缓存
            if cache_key in _data_cache:
                logger.debug(f"从缓存加载股票 {symbol} 数据")
                df = _data_cache[cache_key]
                validated_data[symbol] = df
                continue
            
            # 从存储加载数据
            stock_data = storage.load_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not stock_data or not stock_data.bars:
                missing_symbols.append(symbol)
                logger.warning(f"股票 {symbol} 没有数据")
                continue
            
            # 转换为DataFrame
            df = stock_data.to_dataframe()
            
            if df.empty:
                missing_symbols.append(symbol)
                logger.warning(f"股票 {symbol} 数据为空")
                continue
            
            # 数据质量验证
            validation_result = _validate_stock_data(df, symbol, start_date, end_date)
            
            if not validation_result['is_valid']:
                invalid_symbols.append(symbol)
                logger.warning(f"股票 {symbol} 数据验证失败: {validation_result['message']}")
                continue
            
            # 数据预处理
            df = _preprocess_stock_data(df, symbol)
            
            # 缓存数据
            _cache_data(cache_key, df)
            
            validated_data[symbol] = df
            logger.debug(f"股票 {symbol} 数据加载并验证成功，数据点数: {len(df)}")
            
            # 更新进度
            if task_id and backtest_id:
                progress = 0.3 + (i + 1) / len(symbols) * 0.2  # 0.3-0.5之间
                _backtest_tasks[task_id]["progress"] = progress
                _backtest_tasks[task_id]["current_step"] = f"加载数据 ({i+1}/{len(symbols)})"
                _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
                
                # WebSocket推送详细进度
                await broadcast_backtest_progress(
                    backtest_id=backtest_id,
                    status="running",
                    progress=progress,
                    current_step=f"加载股票数据 ({i+1}/{len(symbols)})"
                )
            
        except Exception as e:
            invalid_symbols.append(symbol)
            logger.error(f"加载股票 {symbol} 数据时出错: {e}")
            continue
    
    # 生成加载报告
    total_symbols = len(symbols)
    loaded_symbols = len(validated_data)
    missing_count = len(missing_symbols)
    invalid_count = len(invalid_symbols)
    
    logger.info(f"数据加载完成: 总数 {total_symbols}, 成功 {loaded_symbols}, 缺失 {missing_count}, 无效 {invalid_count}")
    
    if missing_symbols:
        logger.warning(f"缺失数据的股票: {missing_symbols}")
    
    if invalid_symbols:
        logger.warning(f"数据无效的股票: {invalid_symbols}")
    
    return validated_data


def _validate_stock_data(df: pd.DataFrame, symbol: str, start_date: date, end_date: date) -> Dict[str, Any]:
    """
    验证股票数据质量
    
    Args:
        df: 股票数据DataFrame
        symbol: 股票代码
        start_date: 期望开始日期
        end_date: 期望结束日期
        
    Returns:
        Dict[str, Any]: 验证结果
    """
    try:
        # 检查必需列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                'is_valid': False,
                'message': f"缺少必需列: {missing_columns}"
            }
        
        # 检查数据点数量
        min_data_points = 30  # 至少需要30个数据点
        if len(df) < min_data_points:
            return {
                'is_valid': False,
                'message': f"数据点不足，需要至少{min_data_points}个，实际{len(df)}个"
            }
        
        # 检查日期范围
        df['date'] = pd.to_datetime(df['date'])
        actual_start = df['date'].min().date()
        actual_end = df['date'].max().date()
        
        # 允许一定的日期偏差（考虑到节假日等因素）
        date_tolerance = timedelta(days=30)
        
        if actual_start > start_date + date_tolerance:
            return {
                'is_valid': False,
                'message': f"数据开始日期 {actual_start} 晚于期望日期 {start_date} 过多"
            }
        
        if actual_end < end_date - date_tolerance:
            return {
                'is_valid': False,
                'message': f"数据结束日期 {actual_end} 早于期望日期 {end_date} 过多"
            }
        
        # 检查价格数据的合理性
        for price_col in ['open', 'high', 'low', 'close']:
            if (df[price_col] <= 0).any():
                return {
                    'is_valid': False,
                    'message': f"{price_col}列存在非正数值"
                }
        
        # 检查OHLC逻辑关系
        if ((df['high'] < df['low']) | 
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])).any():
            return {
                'is_valid': False,
                'message': "OHLC数据存在逻辑错误"
            }
        
        # 检查成交量
        if (df['volume'] < 0).any():
            return {
                'is_valid': False,
                'message': "成交量存在负数"
            }
        
        return {
            'is_valid': True,
            'message': "数据验证通过",
            'data_points': len(df),
            'date_range': (actual_start, actual_end)
        }
        
    except Exception as e:
        return {
            'is_valid': False,
            'message': f"数据验证时出错: {str(e)}"
        }


def _preprocess_stock_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    预处理股票数据
    
    Args:
        df: 原始股票数据
        symbol: 股票代码
        
    Returns:
        pd.DataFrame: 预处理后的数据
    """
    try:
        # 复制数据避免修改原始数据
        processed_df = df.copy()
        
        # 确保日期列为datetime类型
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        # 按日期排序
        processed_df = processed_df.sort_values('date').reset_index(drop=True)
        
        # 去除重复日期（保留最后一条记录）
        processed_df = processed_df.drop_duplicates(subset=['date'], keep='last')
        
        # 处理缺失值（前向填充，但限制填充范围）
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in processed_df.columns:
                # 前向填充，但限制最多填充3个连续缺失值
                processed_df[col] = processed_df[col].fillna(method='ffill', limit=3)
        
        # 计算基本技术指标作为预处理的一部分
        processed_df['returns'] = processed_df['close'].pct_change()
        processed_df['log_returns'] = np.log(processed_df['close'] / processed_df['close'].shift(1))
        
        # 计算简单移动平均作为数据平滑
        processed_df['sma_5'] = processed_df['close'].rolling(window=5).mean()
        processed_df['sma_20'] = processed_df['close'].rolling(window=20).mean()
        
        logger.debug(f"股票 {symbol} 数据预处理完成，最终数据点数: {len(processed_df)}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"预处理股票 {symbol} 数据时出错: {e}")
        return df  # 返回原始数据


def _cache_data(cache_key: str, data: pd.DataFrame):
    """
    缓存数据
    
    Args:
        cache_key: 缓存键
        data: 要缓存的数据
    """
    global _data_cache
    
    # 如果缓存已满，删除最旧的条目
    if len(_data_cache) >= _cache_max_size:
        # 删除第一个条目（最旧的）
        oldest_key = next(iter(_data_cache))
        del _data_cache[oldest_key]
        logger.debug(f"缓存已满，删除最旧条目: {oldest_key}")
    
    _data_cache[cache_key] = data.copy()
    logger.debug(f"数据已缓存: {cache_key}, 当前缓存大小: {len(_data_cache)}")
