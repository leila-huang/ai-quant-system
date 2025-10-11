"""
权重管理器

提供权重的验证、调整和监控功能。
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """约束类型"""
    POSITION_LIMIT = "position_limit"      # 单股仓位限制
    INDUSTRY_LIMIT = "industry_limit"      # 行业仓位限制
    SECTOR_LIMIT = "sector_limit"          # 板块仓位限制
    TURNOVER_LIMIT = "turnover_limit"      # 换手率限制
    CONCENTRATION = "concentration"         # 集中度限制


@dataclass
class WeightConstraint:
    """权重约束"""
    constraint_type: ConstraintType
    value: float
    target: Optional[str] = None  # 约束目标（如行业名称）
    
    def __str__(self):
        if self.target:
            return f"{self.constraint_type.value}({self.target}): {self.value}"
        return f"{self.constraint_type.value}: {self.value}"


class WeightManager:
    """
    权重管理器
    
    管理和验证投资组合权重，确保满足所有约束条件。
    """
    
    def __init__(self):
        self.constraints: List[WeightConstraint] = []
        logger.info("权重管理器初始化完成")
    
    def add_constraint(self, constraint: WeightConstraint):
        """添加约束条件"""
        self.constraints.append(constraint)
        logger.debug(f"添加约束: {constraint}")
    
    def validate_weights(self, 
                        weights: pd.Series,
                        industry_map: Optional[Dict[str, str]] = None,
                        current_weights: Optional[pd.Series] = None) -> Dict[str, bool]:
        """
        验证权重是否满足所有约束
        
        Returns:
            Dict[str, bool]: 约束验证结果
        """
        results = {}
        
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.POSITION_LIMIT:
                results['position_limit'] = self._check_position_limit(weights, constraint.value)
            
            elif constraint.constraint_type == ConstraintType.INDUSTRY_LIMIT:
                if industry_map:
                    results['industry_limit'] = self._check_industry_limit(
                        weights, industry_map, constraint.value, constraint.target
                    )
            
            elif constraint.constraint_type == ConstraintType.TURNOVER_LIMIT:
                if current_weights is not None:
                    results['turnover_limit'] = self._check_turnover_limit(
                        weights, current_weights, constraint.value
                    )
            
            elif constraint.constraint_type == ConstraintType.CONCENTRATION:
                results['concentration'] = self._check_concentration(weights, constraint.value)
        
        return results
    
    def _check_position_limit(self, weights: pd.Series, max_weight: float) -> bool:
        """检查单股仓位限制"""
        max_actual = weights.max()
        passed = max_actual <= max_weight
        if not passed:
            logger.warning(f"单股仓位超限: {max_actual:.4f} > {max_weight:.4f}")
        return passed
    
    def _check_industry_limit(self,
                             weights: pd.Series,
                             industry_map: Dict[str, str],
                             max_weight: float,
                             target_industry: Optional[str]) -> bool:
        """检查行业仓位限制"""
        industry_weights = {}
        for symbol in weights.index:
            industry = industry_map.get(symbol, 'Unknown')
            if industry not in industry_weights:
                industry_weights[industry] = 0
            industry_weights[industry] += weights[symbol]
        
        if target_industry:
            # 检查特定行业
            actual = industry_weights.get(target_industry, 0)
            passed = actual <= max_weight
            if not passed:
                logger.warning(f"行业 {target_industry} 仓位超限: {actual:.4f} > {max_weight:.4f}")
            return passed
        else:
            # 检查所有行业
            passed = all(w <= max_weight for w in industry_weights.values())
            if not passed:
                over_limit = {k: v for k, v in industry_weights.items() if v > max_weight}
                logger.warning(f"行业仓位超限: {over_limit}")
            return passed
    
    def _check_turnover_limit(self,
                             weights: pd.Series,
                             current_weights: pd.Series,
                             max_turnover: float) -> bool:
        """检查换手率限制"""
        # 对齐权重
        all_symbols = set(weights.index) | set(current_weights.index)
        aligned_new = pd.Series(0, index=all_symbols)
        aligned_old = pd.Series(0, index=all_symbols)
        
        aligned_new.update(weights)
        aligned_old.update(current_weights)
        
        # 计算换手率
        turnover = np.sum(np.abs(aligned_new - aligned_old))
        passed = turnover <= max_turnover
        
        if not passed:
            logger.warning(f"换手率超限: {turnover:.4f} > {max_turnover:.4f}")
        
        return passed
    
    def _check_concentration(self, weights: pd.Series, max_concentration: float) -> bool:
        """
        检查集中度限制
        
        集中度定义为Top5持仓占比
        """
        top5_weight = weights.nlargest(5).sum()
        passed = top5_weight <= max_concentration
        
        if not passed:
            logger.warning(f"集中度超限: {top5_weight:.4f} > {max_concentration:.4f}")
        
        return passed
    
    def adjust_weights_to_constraints(self,
                                     weights: pd.Series,
                                     industry_map: Optional[Dict[str, str]] = None,
                                     max_iterations: int = 10) -> pd.Series:
        """
        调整权重以满足约束
        
        迭代调整权重直到满足所有约束
        """
        adjusted_weights = weights.copy()
        
        for iteration in range(max_iterations):
            # 检查当前约束
            validation = self.validate_weights(adjusted_weights, industry_map)
            
            # 如果全部通过，返回
            if all(validation.values()):
                logger.info(f"权重调整完成，迭代{iteration+1}次")
                return adjusted_weights
            
            # 应用调整
            for constraint in self.constraints:
                if constraint.constraint_type == ConstraintType.POSITION_LIMIT:
                    adjusted_weights = adjusted_weights.clip(upper=constraint.value)
                
                elif constraint.constraint_type == ConstraintType.INDUSTRY_LIMIT:
                    if industry_map:
                        adjusted_weights = self._adjust_industry_weights(
                            adjusted_weights, industry_map, constraint.value
                        )
            
            # 重新归一化
            total = adjusted_weights.sum()
            if total > 0:
                adjusted_weights = adjusted_weights / total
        
        logger.warning(f"权重调整未完全收敛，已迭代{max_iterations}次")
        return adjusted_weights
    
    def _adjust_industry_weights(self,
                                weights: pd.Series,
                                industry_map: Dict[str, str],
                                max_weight: float) -> pd.Series:
        """调整行业权重至限制内"""
        adjusted = weights.copy()
        
        # 计算当前行业权重
        industry_weights = {}
        industry_stocks = {}
        for symbol in weights.index:
            industry = industry_map.get(symbol, 'Unknown')
            if industry not in industry_weights:
                industry_weights[industry] = 0
                industry_stocks[industry] = []
            industry_weights[industry] += weights[symbol]
            industry_stocks[industry].append(symbol)
        
        # 调整超限行业
        for industry, total_weight in industry_weights.items():
            if total_weight > max_weight:
                # 等比例缩减该行业所有股票
                scale = max_weight / total_weight
                for symbol in industry_stocks[industry]:
                    adjusted[symbol] *= scale
        
        return adjusted
    
    def get_weight_statistics(self,
                             weights: pd.Series,
                             industry_map: Optional[Dict[str, str]] = None) -> Dict:
        """
        获取权重统计信息
        """
        stats = {
            'n_positions': len(weights[weights > 0]),
            'total_weight': weights.sum(),
            'max_weight': weights.max(),
            'min_weight': weights[weights > 0].min() if len(weights[weights > 0]) > 0 else 0,
            'mean_weight': weights[weights > 0].mean() if len(weights[weights > 0]) > 0 else 0,
            'concentration_top5': weights.nlargest(5).sum(),
            'concentration_top10': weights.nlargest(10).sum(),
        }
        
        # 行业分布
        if industry_map:
            industry_weights = {}
            for symbol in weights.index:
                if weights[symbol] > 0:
                    industry = industry_map.get(symbol, 'Unknown')
                    if industry not in industry_weights:
                        industry_weights[industry] = 0
                    industry_weights[industry] += weights[symbol]
            
            stats['industry_distribution'] = industry_weights
            stats['n_industries'] = len(industry_weights)
        
        return stats

