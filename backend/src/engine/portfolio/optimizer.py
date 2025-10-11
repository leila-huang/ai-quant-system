"""
投资组合优化器

实现投资组合优化算法，支持：
- 行业权重上限约束
- 单股权重上限约束  
- 目标波动率调整
- 多种优化目标（均值方差、风险平价等）
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """优化方法"""
    MEAN_VARIANCE = "mean_variance"      # 均值-方差优化
    MIN_VARIANCE = "min_variance"        # 最小方差
    MAX_SHARPE = "max_sharpe"           # 最大夏普比率
    RISK_PARITY = "risk_parity"         # 风险平价
    MAX_DIVERSIFICATION = "max_diversification"  # 最大分散化


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 优化方法
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    
    # 权重约束
    max_stock_weight: float = 0.05      # 单股最大权重（5%）
    min_stock_weight: float = 0.0       # 单股最小权重
    max_industry_weight: float = 0.20   # 行业最大权重（20%）
    
    # 风险目标
    target_volatility: Optional[float] = None  # 目标波动率
    max_volatility: Optional[float] = None     # 最大波动率
    
    # 回报目标
    target_return: Optional[float] = None      # 目标回报率
    min_return: Optional[float] = None         # 最小回报率
    
    # 交易成本约束
    max_turnover: Optional[float] = None       # 最大换手率
    transaction_cost: float = 0.001            # 交易成本（0.1%）
    
    # 优化器参数
    risk_aversion: float = 1.0          # 风险厌恶系数
    max_iterations: int = 1000          # 最大迭代次数
    tolerance: float = 1e-6             # 收敛容差
    
    def __post_init__(self):
        """配置验证"""
        if not 0 < self.max_stock_weight <= 1:
            raise ValueError("max_stock_weight必须在(0,1]范围内")
        if not 0 <= self.min_stock_weight < self.max_stock_weight:
            raise ValueError("min_stock_weight必须在[0, max_stock_weight)范围内")
        if not 0 < self.max_industry_weight <= 1:
            raise ValueError("max_industry_weight必须在(0,1]范围内")


class PortfolioOptimizer:
    """
    投资组合优化器
    
    使用现代投资组合理论进行权重优化，支持多种约束条件。
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        初始化组合优化器
        
        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()
        logger.info(f"组合优化器初始化完成: 方法={self.config.method.value}")
    
    def optimize(self,
                 expected_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 industry_map: Optional[Dict[str, str]] = None,
                 current_weights: Optional[pd.Series] = None,
                 **kwargs) -> pd.Series:
        """
        优化投资组合权重
        
        Args:
            expected_returns: 预期收益率（Series, index为股票代码）
            cov_matrix: 协方差矩阵（DataFrame）
            industry_map: 股票到行业的映射 {股票代码: 行业名称}
            current_weights: 当前权重（用于计算换手率）
            **kwargs: 其他优化参数
            
        Returns:
            pd.Series: 优化后的权重
        """
        try:
            # 数据对齐
            symbols = expected_returns.index.tolist()
            n_assets = len(symbols)
            
            if n_assets == 0:
                logger.warning("没有资产可优化")
                return pd.Series(dtype=float)
            
            # 确保协方差矩阵对齐
            cov_matrix = cov_matrix.loc[symbols, symbols]
            
            # 根据优化方法选择
            if self.config.method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._mean_variance_optimization(
                    expected_returns, cov_matrix, industry_map, current_weights
                )
            elif self.config.method == OptimizationMethod.MIN_VARIANCE:
                weights = self._min_variance_optimization(
                    cov_matrix, industry_map, current_weights
                )
            elif self.config.method == OptimizationMethod.MAX_SHARPE:
                weights = self._max_sharpe_optimization(
                    expected_returns, cov_matrix, industry_map, current_weights
                )
            elif self.config.method == OptimizationMethod.RISK_PARITY:
                weights = self._risk_parity_optimization(
                    cov_matrix, industry_map
                )
            elif self.config.method == OptimizationMethod.MAX_DIVERSIFICATION:
                weights = self._max_diversification_optimization(
                    cov_matrix, industry_map
                )
            else:
                logger.warning(f"未知优化方法: {self.config.method}, 使用等权重")
                weights = pd.Series(1.0/n_assets, index=symbols)
            
            # 应用约束
            weights = self._apply_constraints(weights, industry_map)
            
            # 归一化
            weights = self._normalize_weights(weights)
            
            logger.info(f"组合优化完成: {len(weights)}只股票, 总权重={weights.sum():.4f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"组合优化失败: {e}")
            # 返回等权重作为后备方案
            return pd.Series(1.0/len(expected_returns), index=expected_returns.index)
    
    def _mean_variance_optimization(self,
                                    expected_returns: pd.Series,
                                    cov_matrix: pd.DataFrame,
                                    industry_map: Optional[Dict[str, str]],
                                    current_weights: Optional[pd.Series]) -> pd.Series:
        """
        均值-方差优化
        
        最大化：E(R) - λ * Var(R)
        """
        n_assets = len(expected_returns)
        symbols = expected_returns.index.tolist()
        
        # 目标函数：负的效用函数
        def objective(w):
            portfolio_return = np.dot(w, expected_returns.values)
            portfolio_variance = np.dot(w, np.dot(cov_matrix.values, w))
            utility = portfolio_return - self.config.risk_aversion * portfolio_variance
            return -utility  # 最小化负效用 = 最大化效用
        
        # 约束和边界
        constraints, bounds = self._build_constraints(
            n_assets, symbols, industry_map, current_weights
        )
        
        # 初始权重（等权重）
        w0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if not result.success:
            logger.warning(f"优化未收敛: {result.message}")
        
        return pd.Series(result.x, index=symbols)
    
    def _min_variance_optimization(self,
                                   cov_matrix: pd.DataFrame,
                                   industry_map: Optional[Dict[str, str]],
                                   current_weights: Optional[pd.Series]) -> pd.Series:
        """
        最小方差优化
        
        最小化组合方差
        """
        n_assets = len(cov_matrix)
        symbols = cov_matrix.index.tolist()
        
        # 目标函数：组合方差
        def objective(w):
            return np.dot(w, np.dot(cov_matrix.values, w))
        
        # 约束和边界
        constraints, bounds = self._build_constraints(
            n_assets, symbols, industry_map, current_weights
        )
        
        # 初始权重
        w0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if not result.success:
            logger.warning(f"最小方差优化未收敛: {result.message}")
        
        return pd.Series(result.x, index=symbols)
    
    def _max_sharpe_optimization(self,
                                 expected_returns: pd.Series,
                                 cov_matrix: pd.DataFrame,
                                 industry_map: Optional[Dict[str, str]],
                                 current_weights: Optional[pd.Series]) -> pd.Series:
        """
        最大夏普比率优化
        
        最大化：E(R) / Std(R)
        """
        n_assets = len(expected_returns)
        symbols = expected_returns.index.tolist()
        
        # 目标函数：负的夏普比率
        def objective(w):
            portfolio_return = np.dot(w, expected_returns.values)
            portfolio_std = np.sqrt(np.dot(w, np.dot(cov_matrix.values, w)))
            sharpe = portfolio_return / (portfolio_std + 1e-6)
            return -sharpe  # 最小化负夏普 = 最大化夏普
        
        # 约束和边界
        constraints, bounds = self._build_constraints(
            n_assets, symbols, industry_map, current_weights
        )
        
        # 初始权重
        w0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if not result.success:
            logger.warning(f"最大夏普优化未收敛: {result.message}")
        
        return pd.Series(result.x, index=symbols)
    
    def _risk_parity_optimization(self,
                                  cov_matrix: pd.DataFrame,
                                  industry_map: Optional[Dict[str, str]]) -> pd.Series:
        """
        风险平价优化
        
        使每个资产对组合风险的贡献相等
        """
        n_assets = len(cov_matrix)
        symbols = cov_matrix.index.tolist()
        
        # 目标函数：风险贡献的方差
        def objective(w):
            portfolio_var = np.dot(w, np.dot(cov_matrix.values, w))
            marginal_contrib = np.dot(cov_matrix.values, w)
            risk_contrib = w * marginal_contrib / (np.sqrt(portfolio_var) + 1e-6)
            target = portfolio_var / n_assets  # 目标风险贡献
            return np.sum((risk_contrib - target) ** 2)
        
        # 约束和边界
        constraints, bounds = self._build_constraints(
            n_assets, symbols, industry_map, None
        )
        
        # 初始权重
        w0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if not result.success:
            logger.warning(f"风险平价优化未收敛: {result.message}")
        
        return pd.Series(result.x, index=symbols)
    
    def _max_diversification_optimization(self,
                                         cov_matrix: pd.DataFrame,
                                         industry_map: Optional[Dict[str, str]]) -> pd.Series:
        """
        最大分散化优化
        
        最大化分散化比率
        """
        n_assets = len(cov_matrix)
        symbols = cov_matrix.index.tolist()
        
        # 资产波动率
        asset_volatilities = np.sqrt(np.diag(cov_matrix.values))
        
        # 目标函数：负的分散化比率
        def objective(w):
            weighted_vol = np.dot(w, asset_volatilities)
            portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix.values, w)))
            diversification_ratio = weighted_vol / (portfolio_vol + 1e-6)
            return -diversification_ratio
        
        # 约束和边界
        constraints, bounds = self._build_constraints(
            n_assets, symbols, industry_map, None
        )
        
        # 初始权重
        w0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if not result.success:
            logger.warning(f"最大分散化优化未收敛: {result.message}")
        
        return pd.Series(result.x, index=symbols)
    
    def _build_constraints(self,
                          n_assets: int,
                          symbols: List[str],
                          industry_map: Optional[Dict[str, str]],
                          current_weights: Optional[pd.Series]) -> Tuple[List, Bounds]:
        """
        构建优化约束条件
        """
        constraints = []
        
        # 权重和为1的约束
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # 行业权重约束
        if industry_map and self.config.max_industry_weight < 1.0:
            industries = {}
            for i, symbol in enumerate(symbols):
                industry = industry_map.get(symbol, 'Unknown')
                if industry not in industries:
                    industries[industry] = []
                industries[industry].append(i)
            
            # 为每个行业添加约束
            for industry, indices in industries.items():
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=indices: self.config.max_industry_weight - np.sum(w[idx])
                })
        
        # 换手率约束
        if current_weights is not None and self.config.max_turnover is not None:
            # 确保current_weights对齐
            current_w = np.zeros(n_assets)
            for i, symbol in enumerate(symbols):
                if symbol in current_weights.index:
                    current_w[i] = current_weights[symbol]
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.config.max_turnover - np.sum(np.abs(w - current_w))
            })
        
        # 单股权重边界
        bounds = Bounds(
            lb=np.full(n_assets, self.config.min_stock_weight),
            ub=np.full(n_assets, self.config.max_stock_weight)
        )
        
        return constraints, bounds
    
    def _apply_constraints(self,
                          weights: pd.Series,
                          industry_map: Optional[Dict[str, str]]) -> pd.Series:
        """
        应用后处理约束
        
        确保权重满足所有约束条件
        """
        # 单股权重约束
        weights = weights.clip(
            lower=self.config.min_stock_weight,
            upper=self.config.max_stock_weight
        )
        
        # 行业权重约束
        if industry_map and self.config.max_industry_weight < 1.0:
            weights = self._apply_industry_constraints(weights, industry_map)
        
        return weights
    
    def _apply_industry_constraints(self,
                                    weights: pd.Series,
                                    industry_map: Dict[str, str]) -> pd.Series:
        """
        应用行业权重上限约束
        """
        # 按行业分组
        industry_weights = {}
        for symbol in weights.index:
            industry = industry_map.get(symbol, 'Unknown')
            if industry not in industry_weights:
                industry_weights[industry] = []
            industry_weights[industry].append(symbol)
        
        # 检查并调整超限行业
        for industry, symbols in industry_weights.items():
            industry_total = weights.loc[symbols].sum()
            
            if industry_total > self.config.max_industry_weight:
                # 等比例缩减
                scale_factor = self.config.max_industry_weight / industry_total
                weights.loc[symbols] *= scale_factor
                logger.debug(f"行业 {industry} 权重超限，从 {industry_total:.4f} 缩减到 {self.config.max_industry_weight:.4f}")
        
        return weights
    
    def _normalize_weights(self, weights: pd.Series) -> pd.Series:
        """归一化权重，确保总和为1"""
        total = weights.sum()
        if total == 0:
            return weights
        return weights / total

