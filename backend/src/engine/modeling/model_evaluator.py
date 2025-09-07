"""
模型评估器

提供全面的模型性能评估功能，包括回归和分类指标、
金融领域特定的评估指标和可视化报告。
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from scipy import stats

from backend.src.engine.modeling import ModelEvaluator as BaseModelEvaluator, PredictionTarget

logger = logging.getLogger(__name__)


class ComprehensiveModelEvaluator(BaseModelEvaluator):
    """
    综合模型评估器
    
    提供回归、分类和金融领域特定的评估指标，
    支持模型比较和性能分析报告生成。
    """
    
    def __init__(self):
        """初始化评估器"""
        self.evaluation_history = []
        self.supported_regression_metrics = [
            'mse', 'mae', 'rmse', 'r2', 'mape', 'explained_variance', 
            'max_error', 'median_absolute_error'
        ]
        self.supported_classification_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
            'precision_macro', 'recall_macro', 'f1_macro'
        ]
        self.supported_financial_metrics = [
            'sharpe_ratio', 'information_ratio', 'hit_rate', 'directional_accuracy',
            'max_drawdown', 'calmar_ratio', 'sortino_ratio'
        ]
        
        logger.info("综合模型评估器初始化完成")
    
    def evaluate(self, 
                y_true: np.ndarray, 
                y_pred: np.ndarray, 
                prediction_target: PredictionTarget = PredictionTarget.RETURN,
                **kwargs) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值  
            prediction_target: 预测目标类型
            **kwargs: 额外参数
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        logger.info(f"开始评估模型性能，预测目标: {prediction_target.value}")
        
        if len(y_true) != len(y_pred):
            raise ValueError("真实值和预测值长度不匹配")
        
        if len(y_true) == 0:
            raise ValueError("数据不能为空")
        
        try:
            evaluation_results = {}
            
            # 基础统计信息
            evaluation_results.update(self._calculate_basic_stats(y_true, y_pred))
            
            # 根据预测目标计算对应指标
            if prediction_target in [PredictionTarget.DIRECTION, PredictionTarget.CLASSIFICATION]:
                evaluation_results.update(self._calculate_classification_metrics(y_true, y_pred))
            else:
                evaluation_results.update(self._calculate_regression_metrics(y_true, y_pred))
            
            # 金融领域特定指标
            if prediction_target == PredictionTarget.RETURN:
                evaluation_results.update(self._calculate_financial_metrics(y_true, y_pred, **kwargs))
            
            # 记录评估历史
            self.evaluation_history.append({
                'timestamp': datetime.now(),
                'prediction_target': prediction_target.value,
                'metrics': evaluation_results,
                'sample_count': len(y_true)
            })
            
            logger.info(f"模型评估完成，计算了 {len(evaluation_results)} 个指标")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            raise
    
    def _calculate_basic_stats(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算基础统计信息"""
        residuals = y_pred - y_true
        
        return {
            'sample_count': len(y_true),
            'y_true_mean': np.mean(y_true),
            'y_true_std': np.std(y_true),
            'y_pred_mean': np.mean(y_pred),
            'y_pred_std': np.std(y_pred),
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        }
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算回归评估指标"""
        metrics = {}
        
        try:
            # 基础回归指标
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # 避免除零错误
            if not np.any(y_true == 0):
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            else:
                metrics['mape'] = np.nan
            
            # 其他回归指标
            from sklearn.metrics import explained_variance_score, max_error, median_absolute_error
            
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
            metrics['max_error'] = max_error(y_true, y_pred) 
            metrics['median_absolute_error'] = median_absolute_error(y_true, y_pred)
            
            # 自定义指标
            metrics['mean_bias'] = np.mean(y_pred - y_true)
            
            # 相对指标
            if np.std(y_true) != 0:
                metrics['normalized_rmse'] = metrics['rmse'] / np.std(y_true)
            else:
                metrics['normalized_rmse'] = np.nan
            
        except Exception as e:
            logger.warning(f"计算回归指标时发生错误: {e}")
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算分类评估指标"""
        metrics = {}
        
        try:
            # 基础分类指标
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # 宏观平均指标
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # ROC AUC（仅适用于二分类或有概率输出时）
            unique_classes = np.unique(y_true)
            if len(unique_classes) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except Exception:
                    metrics['roc_auc'] = np.nan
            
            # 混淆矩阵相关指标
            cm = confusion_matrix(y_true, y_pred)
            if cm.size > 0:
                # 对于二分类问题
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
        except Exception as e:
            logger.warning(f"计算分类指标时发生错误: {e}")
        
        return metrics
    
    def _calculate_financial_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray, 
                                   **kwargs) -> Dict[str, float]:
        """计算金融领域特定指标"""
        metrics = {}
        
        try:
            # 获取额外参数
            risk_free_rate = kwargs.get('risk_free_rate', 0.02)  # 默认2%无风险收益率
            benchmark_returns = kwargs.get('benchmark_returns', None)
            
            # 夏普比率（基于预测收益率）
            if np.std(y_pred) != 0:
                metrics['sharpe_ratio'] = (np.mean(y_pred) - risk_free_rate) / np.std(y_pred)
            else:
                metrics['sharpe_ratio'] = 0
            
            # 信息比率（相对于基准）
            if benchmark_returns is not None and len(benchmark_returns) == len(y_pred):
                excess_returns = y_pred - benchmark_returns
                if np.std(excess_returns) != 0:
                    metrics['information_ratio'] = np.mean(excess_returns) / np.std(excess_returns)
                else:
                    metrics['information_ratio'] = 0
            
            # 命中率（预测方向正确的比例）
            pred_direction = np.sign(y_pred)
            true_direction = np.sign(y_true)
            metrics['hit_rate'] = np.mean(pred_direction == true_direction)
            
            # 方向准确率（非零预测的方向准确率）
            non_zero_mask = (y_pred != 0) & (y_true != 0)
            if np.any(non_zero_mask):
                metrics['directional_accuracy'] = np.mean(
                    pred_direction[non_zero_mask] == true_direction[non_zero_mask]
                )
            else:
                metrics['directional_accuracy'] = 0
            
            # 最大回撤（基于累积收益）
            if len(y_pred) > 1:
                cum_returns = np.cumprod(1 + y_pred) - 1
                peak = np.maximum.accumulate(cum_returns)
                drawdown = (cum_returns - peak) / (1 + peak)
                metrics['max_drawdown'] = np.min(drawdown)
            else:
                metrics['max_drawdown'] = 0
            
            # 卡尔玛比率
            if metrics.get('max_drawdown', 0) != 0:
                metrics['calmar_ratio'] = np.mean(y_pred) / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = 0
            
            # Sortino比率（仅考虑下行风险）
            downside_returns = y_pred[y_pred < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) != 0:
                metrics['sortino_ratio'] = (np.mean(y_pred) - risk_free_rate) / np.std(downside_returns)
            else:
                metrics['sortino_ratio'] = 0
            
            # 尾部风险指标
            if len(y_pred) > 0:
                metrics['var_95'] = np.percentile(y_pred, 5)  # 95% VaR
                metrics['cvar_95'] = np.mean(y_pred[y_pred <= metrics['var_95']])  # 95% CVaR
            
        except Exception as e:
            logger.warning(f"计算金融指标时发生错误: {e}")
        
        return metrics
    
    def compare_models(self, 
                      evaluation_results: Dict[str, Dict[str, float]],
                      metrics_to_compare: List[str] = None) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            evaluation_results: 模型评估结果字典
            metrics_to_compare: 要比较的指标列表
            
        Returns:
            Dict[str, Any]: 模型比较结果
        """
        if not evaluation_results:
            raise ValueError("评估结果不能为空")
        
        logger.info(f"开始比较 {len(evaluation_results)} 个模型")
        
        # 默认比较指标
        if metrics_to_compare is None:
            # 从第一个模型的结果中获取所有数值指标
            first_result = next(iter(evaluation_results.values()))
            metrics_to_compare = [k for k, v in first_result.items() 
                                 if isinstance(v, (int, float)) and not np.isnan(v)]
        
        comparison_results = {
            'model_rankings': {},
            'best_model_per_metric': {},
            'metric_statistics': {},
            'summary': {}
        }
        
        # 为每个指标创建排名
        for metric in metrics_to_compare:
            metric_values = {}
            for model_name, results in evaluation_results.items():
                if metric in results and not np.isnan(results[metric]):
                    metric_values[model_name] = results[metric]
            
            if metric_values:
                # 确定是否是越大越好的指标
                higher_better = metric in ['r2', 'accuracy', 'precision', 'recall', 'f1', 
                                         'roc_auc', 'sharpe_ratio', 'information_ratio', 
                                         'hit_rate', 'directional_accuracy', 'calmar_ratio', 'sortino_ratio']
                
                # 排序
                sorted_models = sorted(metric_values.items(), 
                                     key=lambda x: x[1], 
                                     reverse=higher_better)
                
                comparison_results['model_rankings'][metric] = sorted_models
                comparison_results['best_model_per_metric'][metric] = sorted_models[0]
                
                # 指标统计
                values = list(metric_values.values())
                comparison_results['metric_statistics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
        
        # 综合排名（基于排名均值）
        model_names = list(evaluation_results.keys())
        overall_ranks = {name: [] for name in model_names}
        
        for metric, rankings in comparison_results['model_rankings'].items():
            for rank, (model_name, _) in enumerate(rankings):
                overall_ranks[model_name].append(rank + 1)  # 排名从1开始
        
        # 计算平均排名
        average_ranks = {name: np.mean(ranks) for name, ranks in overall_ranks.items()}
        overall_ranking = sorted(average_ranks.items(), key=lambda x: x[1])
        
        comparison_results['summary'] = {
            'overall_ranking': overall_ranking,
            'best_overall_model': overall_ranking[0][0],
            'models_compared': len(model_names),
            'metrics_compared': len(metrics_to_compare),
            'comparison_timestamp': datetime.now()
        }
        
        logger.info(f"模型比较完成，最佳模型: {comparison_results['summary']['best_overall_model']}")
        return comparison_results
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, float],
                                 model_name: str = "Model",
                                 save_path: str = None) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_results: 评估结果
            model_name: 模型名称
            save_path: 保存路径
            
        Returns:
            str: 报告内容
        """
        logger.info(f"生成模型 {model_name} 的评估报告")
        
        report_lines = [
            f"# {model_name} 模型评估报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 评估指标总览",
            ""
        ]
        
        # 按类别组织指标
        categories = {
            "基础统计": ['sample_count', 'correlation', 'y_true_mean', 'y_pred_mean'],
            "回归指标": ['mse', 'mae', 'rmse', 'r2', 'mape'],
            "分类指标": ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            "金融指标": ['sharpe_ratio', 'hit_rate', 'max_drawdown', 'calmar_ratio']
        }
        
        for category, metrics in categories.items():
            category_metrics = {k: v for k, v in evaluation_results.items() if k in metrics}
            if category_metrics:
                report_lines.append(f"### {category}")
                report_lines.append("")
                for metric, value in category_metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"- **{metric}**: {value:.4f}")
                    else:
                        report_lines.append(f"- **{metric}**: {value}")
                report_lines.append("")
        
        # 其他指标
        other_metrics = {k: v for k, v in evaluation_results.items() 
                        if not any(k in cat_metrics for cat_metrics in categories.values())}
        if other_metrics:
            report_lines.append("### 其他指标")
            report_lines.append("")
            for metric, value in other_metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    report_lines.append(f"- **{metric}**: {value}")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"评估报告已保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        return report_content
    
    def plot_model_performance(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             prediction_target: PredictionTarget = PredictionTarget.RETURN,
                             save_path: str = None) -> None:
        """
        绘制模型性能图表
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            prediction_target: 预测目标
            save_path: 保存路径
        """
        plt.style.use('default')
        
        if prediction_target in [PredictionTarget.DIRECTION, PredictionTarget.CLASSIFICATION]:
            # 分类问题的图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # 预测分布
            ax2.hist(y_pred, bins=20, alpha=0.7, label='Predictions')
            ax2.set_title('Prediction Distribution')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Frequency')
            
            # 分类报告（文本）
            ax3.text(0.1, 0.9, "Classification Report:", transform=ax3.transAxes, fontsize=12, fontweight='bold')
            report = classification_report(y_true, y_pred)
            ax3.text(0.1, 0.1, report, transform=ax3.transAxes, fontsize=8, fontfamily='monospace')
            ax3.axis('off')
            
            # 预测准确率随时间变化（如果数据足够）
            if len(y_true) > 50:
                window = len(y_true) // 10
                rolling_accuracy = []
                for i in range(window, len(y_true)):
                    window_acc = accuracy_score(y_true[i-window:i], y_pred[i-window:i])
                    rolling_accuracy.append(window_acc)
                ax4.plot(rolling_accuracy)
                ax4.set_title('Rolling Accuracy')
                ax4.set_xlabel('Time Window')
                ax4.set_ylabel('Accuracy')
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for rolling accuracy', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.axis('off')
        
        else:
            # 回归问题的图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 预测vs实际散点图
            ax1.scatter(y_true, y_pred, alpha=0.6)
            max_val = max(np.max(y_true), np.max(y_pred))
            min_val = min(np.min(y_true), np.min(y_pred))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Predicted vs Actual')
            
            # 残差图
            residuals = y_pred - y_true
            ax2.scatter(y_pred, residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            
            # 残差分布
            ax3.hist(residuals, bins=30, alpha=0.7, density=True)
            ax3.axvline(x=0, color='r', linestyle='--')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Density')
            ax3.set_title('Residual Distribution')
            
            # 时间序列图（如果数据按时间排序）
            ax4.plot(y_true, label='Actual', alpha=0.8)
            ax4.plot(y_pred, label='Predicted', alpha=0.8)
            ax4.legend()
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Values')
            ax4.set_title('Time Series Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"性能图表已保存到: {save_path}")
        
        plt.show()
    
    def get_supported_metrics(self) -> List[str]:
        """获取支持的评估指标"""
        return (self.supported_regression_metrics + 
                self.supported_classification_metrics + 
                self.supported_financial_metrics)
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return self.evaluation_history.copy()


def create_default_evaluator() -> ComprehensiveModelEvaluator:
    """
    创建默认配置的模型评估器
    
    Returns:
        ComprehensiveModelEvaluator: 评估器实例
    """
    return ComprehensiveModelEvaluator()



