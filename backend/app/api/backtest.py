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


router = APIRouter()

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
            cost_model = TradingCostModel(broker_type=broker_type_map.get(request.broker_type, BrokerType.STANDARD))
        
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
        
        # 这里需要实现具体的策略信号生成逻辑
        # 暂时使用模拟信号
        dates = pd.date_range(start=request.start_date, end=request.end_date, freq='D')
        signals_data = {}
        
        for symbol in request.universe[:10]:  # 限制数量以加快演示
            # 生成模拟的买卖信号
            np.random.seed(42)
            signals = pd.Series(
                np.random.choice([0, 1, -1], size=len(dates), p=[0.8, 0.1, 0.1]),
                index=dates,
                name=symbol
            )
            signals_data[symbol] = signals
        
        signals_df = pd.DataFrame(signals_data)
        
        # 更新进度：执行回测
        if task_id:
            _backtest_tasks[task_id]["progress"] = 0.6
            _backtest_tasks[task_id]["current_step"] = "执行策略回测"
            _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 运行回测
        backtest_result = engine.run_backtest(
            strategy_config=request.strategy_config.dict(),
            universe=request.universe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # 更新进度：计算性能指标
        if task_id:
            _backtest_tasks[task_id]["progress"] = 0.8
            _backtest_tasks[task_id]["current_step"] = "计算性能指标"
            _backtest_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"回测执行失败: {str(e)}"
        )
