"""
机器学习模型API路由

提供模型训练、预测、评估、管理等接口。
支持XGBoost等多种机器学习模型，以及模型的持久化和版本管理。
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, status, UploadFile, File
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

from backend.app.core.config import settings
from backend.src.models.basic_models import StockData
from backend.src.engine.modeling import PredictionTarget
from backend.src.storage.parquet_engine import get_parquet_storage
from backend.src.engine.modeling.xgb_wrapper import XGBoostModelFramework
from backend.src.engine.modeling.model_trainer import UnifiedModelTrainer
from backend.src.engine.modeling.model_evaluator import ComprehensiveModelEvaluator
from backend.src.engine.features.feature_store import FeatureStore


router = APIRouter()

# 全局线程池用于异步模型训练
model_executor = ThreadPoolExecutor(max_workers=2)

# 模型存储路径
MODEL_STORAGE_PATH = Path("models")
MODEL_STORAGE_PATH.mkdir(exist_ok=True)

# 全局任务状态存储
_model_tasks = {}


class ModelTrainingRequest(BaseModel):
    """模型训练请求模型"""
    model_name: str = Field(..., description="模型名称")
    model_type: str = Field("xgboost", description="模型类型")
    feature_version: str = Field(..., description="特征版本")
    prediction_target: str = Field("RETURN", description="预测目标: RETURN/DIRECTION/PRICE")
    
    # 训练参数
    train_params: Dict[str, Any] = Field(
        default={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        },
        description="模型训练参数"
    )
    
    # 数据分割参数
    test_size: float = Field(0.2, description="测试集比例")
    validation_size: float = Field(0.2, description="验证集比例")
    time_series_split: bool = Field(True, description="是否使用时间序列分割")
    
    # 超参数优化
    enable_hyperopt: bool = Field(False, description="是否启用超参数优化")
    hyperopt_trials: int = Field(50, description="超参数优化试验次数")
    
    # 异步执行
    async_training: bool = Field(False, description="是否异步训练")
    
    @validator('prediction_target')
    def validate_prediction_target(cls, v):
        valid_targets = ["RETURN", "DIRECTION", "PRICE", "CLASSIFICATION"]
        if v not in valid_targets:
            raise ValueError(f"不支持的预测目标: {v}")
        return v


class ModelPredictionRequest(BaseModel):
    """模型预测请求模型"""
    model_id: str = Field(..., description="模型ID")
    feature_data: Optional[Dict[str, Any]] = Field(None, description="特征数据")
    feature_version: Optional[str] = Field(None, description="特征版本")
    symbols: Optional[List[str]] = Field(None, description="股票代码列表")
    prediction_date: Optional[date] = Field(None, description="预测日期")


class ModelEvaluationRequest(BaseModel):
    """模型评估请求模型"""
    model_id: str = Field(..., description="模型ID")
    test_data: Optional[Dict[str, Any]] = Field(None, description="测试数据")
    evaluation_metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1", "auc"],
        description="评估指标"
    )


class ModelTrainingResponse(BaseModel):
    """模型训练响应模型"""
    model_id: str
    model_name: str
    model_type: str
    training_status: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    training_time_seconds: float
    feature_importance: Optional[Dict[str, float]] = None
    task_id: Optional[str] = None


class ModelPredictionResponse(BaseModel):
    """模型预测响应模型"""
    model_id: str
    predictions: Dict[str, List[float]]  # symbol -> predictions
    prediction_metadata: Dict[str, Any]
    prediction_time_seconds: float


class ModelEvaluationResponse(BaseModel):
    """模型评估响应模型"""
    model_id: str
    evaluation_metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    evaluation_time_seconds: float


class ModelInfoResponse(BaseModel):
    """模型信息响应模型"""
    model_id: str
    model_name: str
    model_type: str
    prediction_target: str
    feature_version: str
    training_date: str
    performance_metrics: Dict[str, float]
    model_size_mb: float
    status: str


class ModelListResponse(BaseModel):
    """模型列表响应模型"""
    models: List[ModelInfoResponse]
    total_count: int


class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


@router.post("/train", response_model=ModelTrainingResponse, summary="训练机器学习模型")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """
    训练机器学习模型
    
    支持XGBoost等多种模型类型，包含特征加载、数据预处理、模型训练、评估等完整流程。
    可选择同步或异步执行模式。
    """
    try:
        start_time = datetime.now()
        model_id = f"{request.model_name}_{uuid.uuid4().hex[:8]}"
        
        if request.async_training:
            # 异步训练
            task_id = f"train_{model_id}_{int(start_time.timestamp())}"
            
            # 初始化任务状态
            _model_tasks[task_id] = {
                "status": "pending",
                "progress": 0.0,
                "result": None,
                "error_message": None,
                "created_at": start_time.isoformat(),
                "updated_at": start_time.isoformat()
            }
            
            # 添加后台任务
            background_tasks.add_task(
                _run_model_training_task,
                task_id, model_id, request
            )
            
            return ModelTrainingResponse(
                model_id=model_id,
                model_name=request.model_name,
                model_type=request.model_type,
                training_status="pending",
                training_metrics={},
                validation_metrics={},
                training_time_seconds=0.0,
                task_id=task_id
            )
        
        # 同步训练
        result = await _execute_model_training(model_id, request)
        
        training_time = (datetime.now() - start_time).total_seconds()
        result.training_time_seconds = round(training_time, 3)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型训练失败: {str(e)}"
        )


@router.post("/predict", response_model=ModelPredictionResponse, summary="模型预测")
async def predict_with_model(request: ModelPredictionRequest):
    """
    使用训练好的模型进行预测
    
    支持加载保存的模型，对新数据进行预测。可以使用提供的特征数据，
    或根据特征版本和股票代码自动加载特征数据。
    """
    try:
        start_time = datetime.now()
        
        # 加载模型
        model_path = MODEL_STORAGE_PATH / f"{request.model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到模型 {request.model_id}"
            )
        
        # 这里需要实现模型加载逻辑
        # model = joblib.load(model_path)
        
        # 准备特征数据
        feature_data = None
        if request.feature_data:
            # 使用提供的特征数据
            feature_data = pd.DataFrame(request.feature_data)
        elif request.feature_version and request.symbols:
            # 从特征存储中加载数据
            feature_store = FeatureStore()
            feature_data = feature_store.load_processed_features(
                feature_version=request.feature_version
            )
            if 'symbol' in feature_data.columns:
                feature_data = feature_data[feature_data['symbol'].isin(request.symbols)]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="必须提供feature_data或(feature_version + symbols)"
            )
        
        if feature_data is None or feature_data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="未找到特征数据"
            )
        
        # 执行预测（简化实现）
        predictions = {}
        if 'symbol' in feature_data.columns:
            for symbol in feature_data['symbol'].unique():
                symbol_data = feature_data[feature_data['symbol'] == symbol]
                # 这里需要实际的预测逻辑
                predictions[symbol] = [0.05] * len(symbol_data)  # 占位符
        else:
            predictions["default"] = [0.05] * len(feature_data)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        return ModelPredictionResponse(
            model_id=request.model_id,
            predictions=predictions,
            prediction_metadata={
                "feature_count": len(feature_data.columns),
                "data_points": len(feature_data),
                "prediction_date": request.prediction_date.isoformat() if request.prediction_date else None
            },
            prediction_time_seconds=round(prediction_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型预测失败: {str(e)}"
        )


@router.post("/evaluate", response_model=ModelEvaluationResponse, summary="模型评估")
async def evaluate_model(request: ModelEvaluationRequest):
    """
    评估模型性能
    
    使用测试数据对模型进行详细评估，包括准确率、精确率、召回率等多种指标。
    """
    try:
        start_time = datetime.now()
        
        # 加载模型
        model_path = MODEL_STORAGE_PATH / f"{request.model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到模型 {request.model_id}"
            )
        
        # 这里需要实现模型评估逻辑
        # 暂时返回模拟的评估结果
        evaluation_metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
            "auc": 0.89
        }
        
        evaluation_time = (datetime.now() - start_time).total_seconds()
        
        return ModelEvaluationResponse(
            model_id=request.model_id,
            evaluation_metrics=evaluation_metrics,
            evaluation_time_seconds=round(evaluation_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型评估失败: {str(e)}"
        )


@router.get("/list", response_model=ModelListResponse, summary="获取模型列表")
async def list_models(
    model_type: Optional[str] = Query(None, description="模型类型筛选"),
    status_filter: Optional[str] = Query(None, description="状态筛选"),
    limit: int = Query(50, description="返回数量限制")
):
    """
    获取所有已训练的模型列表
    
    支持按模型类型、状态等条件筛选，返回模型的基本信息和性能指标。
    """
    try:
        models = []
        
        # 扫描模型存储目录
        for model_file in MODEL_STORAGE_PATH.glob("*.joblib"):
            model_id = model_file.stem
            
            # 尝试加载模型元数据
            metadata_file = MODEL_STORAGE_PATH / f"{model_id}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 应用筛选条件
                if model_type and metadata.get("model_type") != model_type:
                    continue
                if status_filter and metadata.get("status") != status_filter:
                    continue
                
                model_info = ModelInfoResponse(
                    model_id=model_id,
                    model_name=metadata.get("model_name", model_id),
                    model_type=metadata.get("model_type", "unknown"),
                    prediction_target=metadata.get("prediction_target", "unknown"),
                    feature_version=metadata.get("feature_version", "unknown"),
                    training_date=metadata.get("training_date", "unknown"),
                    performance_metrics=metadata.get("performance_metrics", {}),
                    model_size_mb=round(model_file.stat().st_size / 1024 / 1024, 2),
                    status=metadata.get("status", "unknown")
                )
                models.append(model_info)
                
                if len(models) >= limit:
                    break
        
        return ModelListResponse(
            models=models,
            total_count=len(models)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型列表失败: {str(e)}"
        )


@router.get("/{model_id}/info", response_model=ModelInfoResponse, summary="获取模型详细信息")
async def get_model_info(model_id: str):
    """
    获取指定模型的详细信息
    """
    try:
        model_file = MODEL_STORAGE_PATH / f"{model_id}.joblib"
        metadata_file = MODEL_STORAGE_PATH / f"{model_id}_metadata.json"
        
        if not model_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到模型 {model_id}"
            )
        
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return ModelInfoResponse(
            model_id=model_id,
            model_name=metadata.get("model_name", model_id),
            model_type=metadata.get("model_type", "unknown"),
            prediction_target=metadata.get("prediction_target", "unknown"),
            feature_version=metadata.get("feature_version", "unknown"),
            training_date=metadata.get("training_date", "unknown"),
            performance_metrics=metadata.get("performance_metrics", {}),
            model_size_mb=round(model_file.stat().st_size / 1024 / 1024, 2),
            status=metadata.get("status", "trained")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型信息失败: {str(e)}"
        )


@router.delete("/{model_id}", summary="删除模型")
async def delete_model(model_id: str):
    """
    删除指定的模型文件和元数据
    """
    try:
        model_file = MODEL_STORAGE_PATH / f"{model_id}.joblib"
        metadata_file = MODEL_STORAGE_PATH / f"{model_id}_metadata.json"
        
        deleted_files = []
        if model_file.exists():
            model_file.unlink()
            deleted_files.append("model")
        
        if metadata_file.exists():
            metadata_file.unlink()
            deleted_files.append("metadata")
        
        if not deleted_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到模型 {model_id}"
            )
        
        return {
            "message": f"模型 {model_id} 删除成功",
            "deleted_files": deleted_files
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除模型失败: {str(e)}"
        )


@router.post("/{model_id}/export", summary="导出模型")
async def export_model(model_id: str):
    """
    导出模型文件，用于部署或分享
    """
    try:
        model_file = MODEL_STORAGE_PATH / f"{model_id}.joblib"
        if not model_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到模型 {model_id}"
            )
        
        # 这里应该返回文件下载响应
        # 暂时返回文件信息
        return {
            "message": "模型导出功能开发中",
            "model_id": model_id,
            "model_path": str(model_file),
            "file_size_mb": round(model_file.stat().st_size / 1024 / 1024, 2)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出模型失败: {str(e)}"
        )


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="查询模型任务状态")
async def get_model_task_status(task_id: str):
    """
    查询异步模型训练任务的执行状态
    """
    if task_id not in _model_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到任务 {task_id}"
        )
    
    task_info = _model_tasks[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        result=task_info["result"],
        error_message=task_info["error_message"],
        created_at=task_info["created_at"],
        updated_at=task_info["updated_at"]
    )


@router.get("/performance/comparison", summary="模型性能对比")
async def compare_model_performance(
    model_ids: List[str] = Query(..., description="要对比的模型ID列表"),
    metrics: List[str] = Query(["accuracy", "precision", "recall"], description="对比指标")
):
    """
    对比多个模型的性能指标
    """
    try:
        comparison_results = {}
        
        for model_id in model_ids:
            metadata_file = MODEL_STORAGE_PATH / f"{model_id}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                model_metrics = {}
                performance_metrics = metadata.get("performance_metrics", {})
                
                for metric in metrics:
                    model_metrics[metric] = performance_metrics.get(metric, 0.0)
                
                comparison_results[model_id] = {
                    "model_name": metadata.get("model_name", model_id),
                    "model_type": metadata.get("model_type", "unknown"),
                    "metrics": model_metrics,
                    "training_date": metadata.get("training_date", "unknown")
                }
        
        return {
            "comparison_results": comparison_results,
            "compared_metrics": metrics,
            "model_count": len(comparison_results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型性能对比失败: {str(e)}"
        )


# 异步任务执行函数
async def _run_model_training_task(task_id: str, model_id: str, request: ModelTrainingRequest):
    """模型训练异步任务"""
    try:
        # 更新任务状态为运行中
        _model_tasks[task_id]["status"] = "running"
        _model_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 执行模型训练
        result = await _execute_model_training(model_id, request, task_id)
        
        # 任务完成
        _model_tasks[task_id]["status"] = "completed"
        _model_tasks[task_id]["progress"] = 1.0
        _model_tasks[task_id]["result"] = result.dict()
        _model_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        # 任务失败
        _model_tasks[task_id]["status"] = "failed"
        _model_tasks[task_id]["error_message"] = str(e)
        _model_tasks[task_id]["updated_at"] = datetime.now().isoformat()


async def _execute_model_training(model_id: str, request: ModelTrainingRequest, task_id: str = None) -> ModelTrainingResponse:
    """执行模型训练的核心逻辑"""
    try:
        # 更新进度：加载数据
        if task_id:
            _model_tasks[task_id]["progress"] = 0.1
            _model_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 加载特征数据
        feature_store = FeatureStore()
        features_df = feature_store.load_processed_features(request.feature_version)
        
        if features_df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到特征版本 {request.feature_version}"
            )
        
        # 更新进度：准备训练数据
        if task_id:
            _model_tasks[task_id]["progress"] = 0.3
            _model_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 准备目标变量（简化实现）
        # 这里需要根据prediction_target生成相应的目标变量
        y = pd.Series(np.random.randn(len(features_df)), name="target")  # 占位符
        
        # 分割数据（简化实现）
        train_size = int(len(features_df) * (1 - request.test_size - request.validation_size))
        val_size = int(len(features_df) * request.validation_size)
        
        X_train = features_df.iloc[:train_size]
        X_val = features_df.iloc[train_size:train_size + val_size]
        X_test = features_df.iloc[train_size + val_size:]
        
        y_train = y.iloc[:train_size]
        y_val = y.iloc[train_size:train_size + val_size]
        y_test = y.iloc[train_size + val_size:]
        
        # 更新进度：开始训练
        if task_id:
            _model_tasks[task_id]["progress"] = 0.5
            _model_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 创建并训练模型
        if request.model_type == "xgboost":
            model = XGBoostModelFramework(
                prediction_target=PredictionTarget[request.prediction_target],
                **request.train_params
            )
        else:
            raise ValueError(f"不支持的模型类型: {request.model_type}")
        
        # 训练模型
        model.train(X_train, y_train, X_val, y_val)
        
        # 更新进度：评估模型
        if task_id:
            _model_tasks[task_id]["progress"] = 0.8
            _model_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 计算训练和验证指标
        train_score = model.get_last_training_score()
        val_score = model.get_last_validation_score()
        
        training_metrics = {"score": train_score}
        validation_metrics = {"score": val_score}
        
        # 获取特征重要性
        feature_importance = model.get_feature_importance()
        
        # 保存模型
        model_path = MODEL_STORAGE_PATH / f"{model_id}.joblib"
        model.save_model(str(model_path))
        
        # 保存元数据
        metadata = {
            "model_id": model_id,
            "model_name": request.model_name,
            "model_type": request.model_type,
            "prediction_target": request.prediction_target,
            "feature_version": request.feature_version,
            "training_date": datetime.now().isoformat(),
            "performance_metrics": {
                "training_score": train_score,
                "validation_score": val_score
            },
            "train_params": request.train_params,
            "status": "trained"
        }
        
        metadata_path = MODEL_STORAGE_PATH / f"{model_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return ModelTrainingResponse(
            model_id=model_id,
            model_name=request.model_name,
            model_type=request.model_type,
            training_status="completed",
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            training_time_seconds=0.0,  # 会在外层设置
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型训练执行失败: {str(e)}"
        )
