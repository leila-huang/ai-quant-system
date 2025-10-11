"""
组合管理模块

提供投资组合优化、权重管理、信号转换等功能。
支持行业约束、单股权重上限、动态再平衡等高级功能。
"""

from .optimizer import PortfolioOptimizer, OptimizationConfig, OptimizationMethod
from .signal_transformer import SignalToWeightTransformer, SignalTransformConfig
from .weight_manager import WeightManager, WeightConstraint
from .rebalancer import Rebalancer, RebalanceFrequency, RebalanceConfig

__all__ = [
    'PortfolioOptimizer',
    'OptimizationConfig',
    'OptimizationMethod',
    'SignalToWeightTransformer',
    'SignalTransformConfig',
    'WeightManager',
    'WeightConstraint',
    'Rebalancer',
    'RebalanceFrequency',
    'RebalanceConfig',
]

