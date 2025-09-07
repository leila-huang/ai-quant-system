"""
AI量化系统 - P1级核心引擎模块

提供完整的量化研究能力，包括特征工程、机器学习建模、回测引擎等核心功能。
"""

from .features import *
from .modeling import *
from .backtest import *
from .utils import *

__version__ = "0.1.0"
__author__ = "AI Development Team"

# 模块版本信息
ENGINE_VERSION = "0.1.0"

# 支持的功能清单
SUPPORTED_FEATURES = [
    "technical_indicators",    # 技术指标计算
    "feature_engineering",     # 特征工程
    "xgboost_modeling",        # XGBoost建模
    "vectorbt_backtest",       # vectorbt回测
    "market_constraints",      # A股市场约束
    "cost_modeling",          # 交易成本建模
    "performance_reporting"    # 性能报告生成
]

__all__ = [
    "ENGINE_VERSION",
    "SUPPORTED_FEATURES",
    # 子模块会通过import *导入具体类和函数
]



