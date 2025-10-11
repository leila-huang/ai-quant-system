"""
组合管理API路由

提供投资组合优化、权重管理、再平衡等功能的API接口。
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

from backend.src.engine.portfolio.optimizer import (
    PortfolioOptimizer, OptimizationConfig, OptimizationMethod
)
from backend.src.engine.portfolio.signal_transformer import (
    SignalToWeightTransformer, SignalTransformConfig, WeightingMethod
)
from backend.src.engine.portfolio.weight_manager import (
    WeightManager, WeightConstraint, ConstraintType
)
from backend.src.engine.portfolio.rebalancer import (
    Rebalancer, RebalanceConfig, RebalanceFrequency, RebalanceTrigger
)
from backend.app.core.exceptions import APIException

router = APIRouter()
logger = logging.getLogger(__name__)


# === Pydantic模型定义 ===

class SignalTransformRequest(BaseModel):
    """信号转权重请求"""
    signals: Dict[str, float] = Field(..., description="股票信号字典 {股票代码: 信号值}")
    long_quantile: float = Field(0.8, description="多头分位数")
    short_quantile: float = Field(0.2, description="空头分位数")
    weighting_method: str = Field("signal_proportional", description="权重方法")
    max_position_weight: float = Field(0.05, description="单股最大权重")
    
    @validator('weighting_method')
    def validate_method(cls, v):
        valid_methods = [m.value for m in WeightingMethod]
        if v not in valid_methods:
            raise ValueError(f"weighting_method必须是: {', '.join(valid_methods)}")
        return v


class SignalTransformResponse(BaseModel):
    """信号转权重响应"""
    weights: Dict[str, float] = Field(..., description="权重字典")
    n_positions: int = Field(..., description="持仓数量")
    total_weight: float = Field(..., description="总权重")
    max_weight: float = Field(..., description="最大单股权重")


class OptimizePortfolioRequest(BaseModel):
    """组合优化请求"""
    expected_returns: Dict[str, float] = Field(..., description="预期收益率")
    covariance_matrix: List[List[float]] = Field(..., description="协方差矩阵")
    symbols: List[str] = Field(..., description="股票代码列表")
    industry_map: Optional[Dict[str, str]] = Field(None, description="行业映射")
    optimization_method: str = Field("mean_variance", description="优化方法")
    max_stock_weight: float = Field(0.05, description="单股最大权重")
    max_industry_weight: float = Field(0.20, description="行业最大权重")
    risk_aversion: float = Field(1.0, description="风险厌恶系数")


class OptimizePortfolioResponse(BaseModel):
    """组合优化响应"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    n_positions: int


class ValidateWeightsRequest(BaseModel):
    """权重验证请求"""
    weights: Dict[str, float]
    industry_map: Optional[Dict[str, str]] = None
    current_weights: Optional[Dict[str, float]] = None
    constraints: List[Dict[str, Any]] = Field(default_factory=list)


class ValidateWeightsResponse(BaseModel):
    """权重验证响应"""
    is_valid: bool
    validation_results: Dict[str, bool]
    violations: List[str]
    statistics: Dict[str, Any]


class RebalanceRequest(BaseModel):
    """再平衡请求"""
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    portfolio_value: float = Field(1000000.0, description="组合总价值")
    frequency: str = Field("weekly", description="再平衡频率")
    consider_cost: bool = Field(True, description="是否考虑交易成本")


class RebalanceResponse(BaseModel):
    """再平衡响应"""
    should_rebalance: bool
    trades: List[Dict[str, Any]]
    total_turnover: float
    total_cost: float
    drift_before: float


# === API路由 ===

@router.post("/signals/transform", response_model=SignalTransformResponse, summary="信号转权重")
async def transform_signals_to_weights(request: SignalTransformRequest):
    """
    将模型信号转换为投资组合权重
    
    支持横截面分位数选股和多种权重分配方法。
    """
    try:
        # 创建配置
        config = SignalTransformConfig(
            long_quantile=request.long_quantile,
            short_quantile=request.short_quantile,
            weighting_method=WeightingMethod(request.weighting_method),
            max_position_weight=request.max_position_weight
        )
        
        # 创建转换器
        transformer = SignalToWeightTransformer(config)
        
        # 转换信号
        signals = pd.Series(request.signals)
        weights = transformer.transform(signals)
        
        # 构建响应
        return SignalTransformResponse(
            weights=weights.to_dict(),
            n_positions=len(weights[weights > 0]),
            total_weight=float(weights.sum()),
            max_weight=float(weights.max()) if len(weights) > 0 else 0
        )
        
    except Exception as e:
        logger.error(f"信号转权重失败: {e}")
        raise APIException(f"信号转权重失败: {str(e)}")


@router.post("/optimize", response_model=OptimizePortfolioResponse, summary="组合优化")
async def optimize_portfolio(request: OptimizePortfolioRequest):
    """
    优化投资组合权重
    
    使用现代投资组合理论，支持多种优化方法和约束条件。
    """
    try:
        # 创建配置
        config = OptimizationConfig(
            method=OptimizationMethod(request.optimization_method),
            max_stock_weight=request.max_stock_weight,
            max_industry_weight=request.max_industry_weight,
            risk_aversion=request.risk_aversion
        )
        
        # 创建优化器
        optimizer = PortfolioOptimizer(config)
        
        # 准备数据
        expected_returns = pd.Series(request.expected_returns)
        cov_matrix = pd.DataFrame(
            request.covariance_matrix,
            index=request.symbols,
            columns=request.symbols
        )
        
        # 优化
        weights = optimizer.optimize(
            expected_returns,
            cov_matrix,
            industry_map=request.industry_map
        )
        
        # 计算组合指标
        portfolio_return = np.dot(weights.values, expected_returns.loc[weights.index].values)
        portfolio_variance = np.dot(weights.values, np.dot(cov_matrix.loc[weights.index, weights.index].values, weights.values))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return OptimizePortfolioResponse(
            weights=weights.to_dict(),
            expected_return=float(portfolio_return),
            expected_volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe_ratio),
            n_positions=len(weights[weights > 0])
        )
        
    except Exception as e:
        logger.error(f"组合优化失败: {e}")
        raise APIException(f"组合优化失败: {str(e)}")


@router.post("/validate", response_model=ValidateWeightsResponse, summary="权重验证")
async def validate_weights(request: ValidateWeightsRequest):
    """
    验证投资组合权重是否满足约束条件
    """
    try:
        # 创建权重管理器
        manager = WeightManager()
        
        # 添加约束
        for constraint_dict in request.constraints:
            constraint = WeightConstraint(
                constraint_type=ConstraintType(constraint_dict['type']),
                value=constraint_dict['value'],
                target=constraint_dict.get('target')
            )
            manager.add_constraint(constraint)
        
        # 验证权重
        weights = pd.Series(request.weights)
        current_weights = pd.Series(request.current_weights) if request.current_weights else None
        
        validation_results = manager.validate_weights(
            weights,
            industry_map=request.industry_map,
            current_weights=current_weights
        )
        
        # 收集违规项
        violations = [
            f"{key}: 不满足" for key, passed in validation_results.items() if not passed
        ]
        
        # 获取统计信息
        statistics = manager.get_weight_statistics(weights, request.industry_map)
        
        return ValidateWeightsResponse(
            is_valid=all(validation_results.values()),
            validation_results=validation_results,
            violations=violations,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"权重验证失败: {e}")
        raise APIException(f"权重验证失败: {str(e)}")


@router.post("/rebalance", response_model=RebalanceResponse, summary="执行再平衡")
async def execute_rebalance(request: RebalanceRequest):
    """
    执行投资组合再平衡
    
    计算需要执行的交易，考虑交易成本和市场冲击。
    """
    try:
        # 创建配置
        config = RebalanceConfig(
            frequency=RebalanceFrequency(request.frequency),
            consider_cost=request.consider_cost
        )
        
        # 创建再平衡器
        rebalancer = Rebalancer(config)
        
        # 准备权重数据
        current_weights = pd.Series(request.current_weights)
        target_weights = pd.Series(request.target_weights)
        
        # 检查是否需要再平衡
        should_rebalance = rebalancer.should_rebalance(
            datetime.now(),
            current_weights,
            target_weights
        )
        
        if not should_rebalance:
            return RebalanceResponse(
                should_rebalance=False,
                trades=[],
                total_turnover=0,
                total_cost=0,
                drift_before=0
            )
        
        # 计算交易
        trades_df = rebalancer.calculate_trades(
            current_weights,
            target_weights,
            request.portfolio_value
        )
        
        # 转换为字典列表
        trades_list = trades_df.to_dict('records') if len(trades_df) > 0 else []
        
        # 计算统计信息
        total_turnover = trades_df['abs_value'].sum() if len(trades_df) > 0 else 0
        total_cost = trades_df['transaction_cost'].sum() if 'transaction_cost' in trades_df.columns else 0
        drift = rebalancer._calculate_drift(current_weights, target_weights)
        
        return RebalanceResponse(
            should_rebalance=True,
            trades=trades_list,
            total_turnover=float(total_turnover),
            total_cost=float(total_cost),
            drift_before=float(drift)
        )
        
    except Exception as e:
        logger.error(f"再平衡执行失败: {e}")
        raise APIException(f"再平衡执行失败: {str(e)}")


@router.get("/constraints/templates", summary="获取约束模板")
async def get_constraint_templates():
    """
    获取常用约束条件模板
    
    返回预定义的约束配置，便于快速使用。
    """
    templates = {
        "conservative": {
            "name": "保守型",
            "constraints": [
                {"type": "position_limit", "value": 0.03, "description": "单股最大3%"},
                {"type": "industry_limit", "value": 0.15, "description": "行业最大15%"},
                {"type": "concentration", "value": 0.20, "description": "Top5最大20%"},
            ]
        },
        "balanced": {
            "name": "平衡型",
            "constraints": [
                {"type": "position_limit", "value": 0.05, "description": "单股最大5%"},
                {"type": "industry_limit", "value": 0.20, "description": "行业最大20%"},
                {"type": "concentration", "value": 0.30, "description": "Top5最大30%"},
            ]
        },
        "aggressive": {
            "name": "激进型",
            "constraints": [
                {"type": "position_limit", "value": 0.10, "description": "单股最大10%"},
                {"type": "industry_limit", "value": 0.30, "description": "行业最大30%"},
                {"type": "concentration", "value": 0.50, "description": "Top5最大50%"},
            ]
        }
    }
    
    return templates

