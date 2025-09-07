"""
机器学习建模模块

提供完整的机器学习建模能力，包括XGBoost模型训练、预测、
评估和选择等功能，支持多种预测目标。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np


class PredictionTarget(Enum):
    """预测目标类型枚举"""
    PRICE = "price"                    # 价格预测
    RETURN = "return"                  # 收益率预测
    DIRECTION = "direction"            # 涨跌方向预测
    VOLATILITY = "volatility"          # 波动率预测
    CLASSIFICATION = "classification"   # 分类预测


class ModelTrainer(ABC):
    """模型训练器抽象基类"""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ModelTrainer':
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 额外参数
            
        Returns:
            self: 返回自身以支持链式调用
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测结果
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            Dict[str, float]: 特征重要性字典
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """
        保存模型
        
        Args:
            path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            bool: 是否加载成功
        """
        pass


class ModelEvaluator(ABC):
    """模型评估器抽象基类"""
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            **kwargs: 额外参数
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        pass
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """
        获取支持的评估指标
        
        Returns:
            List[str]: 支持的指标列表
        """
        pass


class HyperparameterOptimizer(ABC):
    """超参数优化器抽象基类"""
    
    @abstractmethod
    def optimize(self, X: pd.DataFrame, y: pd.Series, 
                param_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        优化超参数
        
        Args:
            X: 特征数据
            y: 目标变量
            param_space: 参数空间
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 最优参数
        """
        pass


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.models: Dict[str, ModelTrainer] = {}
        self.evaluators: Dict[str, ModelEvaluator] = {}
        
    def register_model(self, name: str, model: ModelTrainer) -> None:
        """注册模型"""
        self.models[name] = model
        
    def register_evaluator(self, name: str, evaluator: ModelEvaluator) -> None:
        """注册评估器"""
        self.evaluators[name] = evaluator
        
    def get_model(self, name: str) -> Optional[ModelTrainer]:
        """获取模型"""
        return self.models.get(name)
        
    def get_evaluator(self, name: str) -> Optional[ModelEvaluator]:
        """获取评估器"""
        return self.evaluators.get(name)


# 模块级常量
SUPPORTED_MODELS = [
    "XGBoost",
    "LightGBM", 
    "RandomForest",
    "LinearRegression",
    "LogisticRegression",
]

SUPPORTED_METRICS = {
    "regression": ["mse", "mae", "rmse", "r2", "mape"],
    "classification": ["accuracy", "precision", "recall", "f1", "auc"],
}

__all__ = [
    "PredictionTarget",
    "ModelTrainer",
    "ModelEvaluator", 
    "HyperparameterOptimizer",
    "ModelManager",
    "SUPPORTED_MODELS",
    "SUPPORTED_METRICS",
]



