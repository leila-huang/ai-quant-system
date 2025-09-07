"""
特征工程模块

提供技术指标计算、特征生成、特征选择等功能，支持从原始股票数据
到机器学习特征的完整转换过程。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

from backend.src.models.basic_models import StockData


class FeatureCalculator(ABC):
    """特征计算器抽象基类"""
    
    @abstractmethod
    def calculate(self, data: Union[StockData, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        计算特征
        
        Args:
            data: 输入数据（StockData对象或DataFrame）
            **kwargs: 额外参数
            
        Returns:
            pd.DataFrame: 包含特征的DataFrame
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            List[str]: 特征名称列表
        """
        pass


class TechnicalIndicatorCalculator(FeatureCalculator):
    """技术指标计算器抽象类"""
    
    @abstractmethod
    def get_supported_indicators(self) -> List[str]:
        """
        获取支持的技术指标列表
        
        Returns:
            List[str]: 支持的指标名称列表
        """
        pass


class FeaturePipeline(ABC):
    """特征工程流水线抽象基类"""
    
    @abstractmethod
    def fit(self, data: Union[StockData, pd.DataFrame]) -> 'FeaturePipeline':
        """
        训练特征流水线
        
        Args:
            data: 训练数据
            
        Returns:
            self: 返回自身以支持链式调用
        """
        pass
    
    @abstractmethod
    def transform(self, data: Union[StockData, pd.DataFrame]) -> pd.DataFrame:
        """
        应用特征转换
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 转换后的特征数据
        """
        pass
    
    def fit_transform(self, data: Union[StockData, pd.DataFrame]) -> pd.DataFrame:
        """
        训练并应用特征转换
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 转换后的特征数据
        """
        return self.fit(data).transform(data)


# 模块级常量
SUPPORTED_TECHNICAL_INDICATORS = [
    "MA",           # 移动平均线
    "EMA",          # 指数移动平均线
    "RSI",          # 相对强弱指标
    "MACD",         # MACD指标
    "BOLL",         # 布林带
    "KDJ",          # KDJ指标
    "WILLIAMS",     # 威廉指标
    "VOLUME_MA",    # 成交量移动平均
]

__all__ = [
    "FeatureCalculator",
    "TechnicalIndicatorCalculator", 
    "FeaturePipeline",
    "SUPPORTED_TECHNICAL_INDICATORS",
]



