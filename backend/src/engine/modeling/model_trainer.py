"""
通用模型训练器

提供统一的模型训练接口和工具，支持多种机器学习算法
和金融时间序列数据的特殊处理需求。
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from backend.src.models.basic_models import StockData
from backend.src.engine.modeling import ModelTrainer as BaseModelTrainer, PredictionTarget
from backend.src.engine.modeling.xgb_wrapper import XGBoostModelFramework
from backend.src.engine.features.pipeline import FeaturePipeline
from backend.src.engine.utils import PerformanceMonitor, ValidationHelper

logger = logging.getLogger(__name__)


class TrainingStrategy(Enum):
    """训练策略枚举"""
    SINGLE_SPLIT = "single_split"              # 单次分割
    TIME_SERIES_CV = "time_series_cv"          # 时间序列交叉验证
    ROLLING_WINDOW = "rolling_window"          # 滚动窗口
    EXPANDING_WINDOW = "expanding_window"      # 扩展窗口


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据分割配置
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 时间窗口配置
    lookback_window: int = 60          # 回看窗口天数
    prediction_horizon: int = 1        # 预测时间范围
    min_train_samples: int = 100       # 最小训练样本数
    
    # 训练策略
    strategy: TrainingStrategy = TrainingStrategy.TIME_SERIES_CV
    cv_folds: int = 5
    
    # 超参数优化
    enable_hyperopt: bool = True
    hyperopt_trials: int = 50
    
    # 其他配置
    random_state: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        # 验证比例配置
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"训练、验证、测试比例之和必须为1，当前为: {total_ratio}")


class UnifiedModelTrainer:
    """
    统一模型训练器
    
    提供完整的模型训练流程管理，包括数据准备、模型训练、
    验证和评估，专门针对金融数据进行优化。
    """
    
    def __init__(self, 
                 config: TrainingConfig = None,
                 feature_pipeline: FeaturePipeline = None):
        """
        初始化统一模型训练器
        
        Args:
            config: 训练配置
            feature_pipeline: 特征工程流水线
        """
        self.config = config or TrainingConfig()
        self.feature_pipeline = feature_pipeline
        
        # 训练状态
        self.is_prepared = False
        self.trained_models = {}
        self.training_results = {}
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("统一模型训练器初始化完成")
    
    def prepare_data(self, 
                    data: Union[StockData, List[StockData], pd.DataFrame],
                    target_column: str = None) -> Dict[str, Any]:
        """
        准备训练数据
        
        Args:
            data: 输入数据
            target_column: 目标列名
            
        Returns:
            Dict[str, Any]: 数据准备结果
        """
        logger.info("开始准备训练数据...")
        
        try:
            # 特征工程
            if self.feature_pipeline:
                if isinstance(data, (StockData, list)):
                    features_df = self.feature_pipeline.transform(data)
                else:
                    features_df = data
            else:
                if isinstance(data, StockData):
                    features_df = data.to_dataframe()
                elif isinstance(data, list) and all(isinstance(d, StockData) for d in data):
                    dfs = [d.to_dataframe() for d in data]
                    features_df = pd.concat(dfs, ignore_index=True)
                else:
                    features_df = data
            
            if features_df.empty:
                raise ValueError("特征数据为空")
            
            # 数据质量检查
            self._validate_data_quality(features_df)
            
            # 按时间排序
            if 'date' in features_df.columns:
                features_df = features_df.sort_values('date').reset_index(drop=True)
            
            # 数据分割
            data_splits = self._split_data(features_df)
            
            self.data_info = {
                'total_samples': len(features_df),
                'feature_count': len(features_df.select_dtypes(include=[np.number]).columns),
                'date_range': self._get_date_range(features_df),
                'splits': {k: len(v) for k, v in data_splits.items()},
                'data_splits': data_splits
            }
            
            self.is_prepared = True
            
            logger.info(f"数据准备完成，总样本: {self.data_info['total_samples']}, 特征数: {self.data_info['feature_count']}")
            return self.data_info
            
        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            raise
    
    def train_model(self, 
                   model: BaseModelTrainer,
                   model_name: str,
                   prediction_target: PredictionTarget = PredictionTarget.RETURN,
                   target_column: str = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            model: 模型实例
            model_name: 模型名称
            prediction_target: 预测目标
            target_column: 目标列名
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        if not self.is_prepared:
            raise ValueError("数据尚未准备，请先调用prepare_data()")
        
        logger.info(f"开始训练模型: {model_name}")
        start_time = datetime.now()
        
        try:
            # 获取数据分割
            data_splits = self.data_info['data_splits']
            train_df = data_splits['train']
            val_df = data_splits['validation']
            test_df = data_splits['test']
            
            # 准备训练数据
            if isinstance(model, XGBoostModelFramework):
                X_train, y_train = model.prepare_training_data(
                    train_df, 
                    target_column=target_column,
                    lookback_window=self.config.lookback_window,
                    prediction_horizon=self.config.prediction_horizon
                )
                X_val, y_val = model.prepare_training_data(
                    val_df,
                    target_column=target_column, 
                    lookback_window=self.config.lookback_window,
                    prediction_horizon=self.config.prediction_horizon
                )
            else:
                # 通用数据准备
                X_train, y_train = self._prepare_generic_data(train_df, target_column, prediction_target)
                X_val, y_val = self._prepare_generic_data(val_df, target_column, prediction_target)
            
            # 超参数优化
            if self.config.enable_hyperopt and hasattr(model, 'optimize_hyperparameters'):
                logger.info("开始超参数优化...")
                best_params = model.optimize_hyperparameters(X_train, y_train)
                logger.info(f"超参数优化完成: {best_params}")
            
            # 训练模型
            model.train(X_train, y_train)
            
            # 验证预测
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # 计算评估指标
            training_results = self._evaluate_predictions(
                y_train, train_pred, y_val, val_pred, prediction_target, model_name
            )
            
            # 保存训练结果
            training_results.update({
                'model_name': model_name,
                'prediction_target': prediction_target.value,
                'training_time': (datetime.now() - start_time).total_seconds(),
                'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
                'feature_importance': model.get_feature_importance() if hasattr(model, 'get_feature_importance') else {},
                'data_info': {
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'feature_count': len(X_train.columns)
                }
            })
            
            self.trained_models[model_name] = model
            self.training_results[model_name] = training_results
            
            logger.info(f"模型 {model_name} 训练完成，训练时间: {training_results['training_time']:.2f}秒")
            return training_results
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def batch_train_models(self, 
                         model_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        批量训练多个模型
        
        Args:
            model_configs: 模型配置列表
            
        Returns:
            Dict[str, Dict[str, Any]]: 所有模型的训练结果
        """
        logger.info(f"开始批量训练 {len(model_configs)} 个模型...")
        
        all_results = {}
        
        for i, config in enumerate(model_configs):
            try:
                model_name = config.get('name', f'model_{i+1}')
                model_class = config.get('model_class', XGBoostModelFramework)
                model_params = config.get('model_params', {})
                prediction_target = config.get('prediction_target', PredictionTarget.RETURN)
                
                # 创建模型实例
                model = model_class(**model_params)
                
                # 训练模型
                result = self.train_model(model, model_name, prediction_target)
                all_results[model_name] = result
                
                logger.info(f"模型 {model_name} 训练完成 ({i+1}/{len(model_configs)})")
                
            except Exception as e:
                logger.error(f"模型 {model_name} 训练失败: {e}")
                all_results[model_name] = {'error': str(e)}
        
        return all_results
    
    def cross_validate_model(self, 
                           model: BaseModelTrainer,
                           prediction_target: PredictionTarget = PredictionTarget.RETURN) -> Dict[str, Any]:
        """
        交叉验证模型
        
        Args:
            model: 模型实例
            prediction_target: 预测目标
            
        Returns:
            Dict[str, Any]: 交叉验证结果
        """
        if not self.is_prepared:
            raise ValueError("数据尚未准备")
        
        logger.info("开始交叉验证...")
        
        # 获取完整数据
        full_data = pd.concat([
            self.data_info['data_splits']['train'],
            self.data_info['data_splits']['validation']
        ], ignore_index=True)
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(full_data)):
            try:
                train_fold = full_data.iloc[train_idx]
                val_fold = full_data.iloc[val_idx]
                
                # 准备数据
                if isinstance(model, XGBoostModelFramework):
                    X_train, y_train = model.prepare_training_data(train_fold)
                    X_val, y_val = model.prepare_training_data(val_fold)
                else:
                    X_train, y_train = self._prepare_generic_data(train_fold, None, prediction_target)
                    X_val, y_val = self._prepare_generic_data(val_fold, None, prediction_target)
                
                # 训练模型
                fold_model = type(model)(**model.model_params if hasattr(model, 'model_params') else {})
                fold_model.train(X_train, y_train)
                
                # 验证预测
                val_pred = fold_model.predict(X_val)
                
                # 计算分数
                score = self._calculate_fold_score(y_val, val_pred, prediction_target)
                cv_scores.append(score)
                
                logger.info(f"折 {fold+1}/{self.config.cv_folds} 完成，分数: {score:.4f}")
                
            except Exception as e:
                logger.error(f"交叉验证折 {fold+1} 失败: {e}")
                cv_scores.append(np.nan)
        
        # 计算交叉验证统计
        cv_scores = np.array(cv_scores)
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': np.nanmean(cv_scores),
            'std_score': np.nanstd(cv_scores),
            'min_score': np.nanmin(cv_scores),
            'max_score': np.nanmax(cv_scores),
            'n_folds': self.config.cv_folds,
            'successful_folds': np.sum(~np.isnan(cv_scores))
        }
        
        logger.info(f"交叉验证完成，平均分数: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        return cv_results
    
    def _split_data(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """数据分割"""
        total_samples = len(features_df)
        
        if self.config.strategy == TrainingStrategy.SINGLE_SPLIT:
            # 单次时间序列分割
            train_end = int(total_samples * self.config.train_ratio)
            val_end = int(total_samples * (self.config.train_ratio + self.config.validation_ratio))
            
            return {
                'train': features_df.iloc[:train_end].copy(),
                'validation': features_df.iloc[train_end:val_end].copy(),
                'test': features_df.iloc[val_end:].copy()
            }
        else:
            # 其他策略的实现（时间序列CV、滚动窗口等）
            # 这里先实现简单版本
            return self._simple_time_split(features_df)
    
    def _simple_time_split(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """简单时间分割"""
        total_samples = len(features_df)
        train_end = int(total_samples * self.config.train_ratio)
        val_end = int(total_samples * (self.config.train_ratio + self.config.validation_ratio))
        
        return {
            'train': features_df.iloc[:train_end].copy(),
            'validation': features_df.iloc[train_end:val_end].copy(), 
            'test': features_df.iloc[val_end:].copy()
        }
    
    def _prepare_generic_data(self, 
                            df: pd.DataFrame, 
                            target_column: str, 
                            prediction_target: PredictionTarget) -> Tuple[pd.DataFrame, pd.Series]:
        """准备通用格式的训练数据"""
        # 选择特征列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['date', 'symbol', 'name']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # 生成目标变量
        if target_column and target_column in df.columns:
            y = df[target_column]
        else:
            if 'close' in df.columns:
                if prediction_target == PredictionTarget.RETURN:
                    y = df['close'].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
                elif prediction_target == PredictionTarget.DIRECTION:
                    future_close = df['close'].shift(-self.config.prediction_horizon)
                    y = (future_close > df['close']).astype(int)
                else:
                    y = df['close'].shift(-self.config.prediction_horizon)
            else:
                raise ValueError("缺少目标变量或close列")
        
        # 移除NaN
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        return X[valid_mask], y[valid_mask]
    
    def _evaluate_predictions(self, 
                            y_train, train_pred, 
                            y_val, val_pred, 
                            prediction_target: PredictionTarget,
                            model_name: str) -> Dict[str, Any]:
        """评估预测结果"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results = {
            'train_metrics': {},
            'validation_metrics': {}
        }
        
        if prediction_target in [PredictionTarget.DIRECTION, PredictionTarget.CLASSIFICATION]:
            # 分类指标
            results['train_metrics'] = {
                'accuracy': accuracy_score(y_train, train_pred),
                'precision': precision_score(y_train, train_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_train, train_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_train, train_pred, average='weighted', zero_division=0)
            }
            
            results['validation_metrics'] = {
                'accuracy': accuracy_score(y_val, val_pred),
                'precision': precision_score(y_val, val_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, val_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_val, val_pred, average='weighted', zero_division=0)
            }
        else:
            # 回归指标
            results['train_metrics'] = {
                'mse': mean_squared_error(y_train, train_pred),
                'mae': mean_absolute_error(y_train, train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'r2': r2_score(y_train, train_pred)
            }
            
            results['validation_metrics'] = {
                'mse': mean_squared_error(y_val, val_pred),
                'mae': mean_absolute_error(y_val, val_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'r2': r2_score(y_val, val_pred)
            }
        
        return results
    
    def _calculate_fold_score(self, y_true, y_pred, prediction_target: PredictionTarget) -> float:
        """计算交叉验证分数"""
        from sklearn.metrics import accuracy_score, r2_score
        
        if prediction_target in [PredictionTarget.DIRECTION, PredictionTarget.CLASSIFICATION]:
            return accuracy_score(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)
    
    def _validate_data_quality(self, df: pd.DataFrame):
        """验证数据质量"""
        if df.empty:
            raise ValueError("数据为空")
        
        # 检查最小样本数
        if len(df) < self.config.min_train_samples:
            raise ValueError(f"样本数不足，需要至少 {self.config.min_train_samples} 个样本，当前: {len(df)}")
        
        # 检查数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("没有找到数值特征")
        
        # 检查缺失值比例
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.5:
            logger.warning(f"数据缺失值比例较高: {missing_ratio:.2%}")
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取日期范围信息"""
        if 'date' not in df.columns:
            return {'has_date': False}
        
        dates = pd.to_datetime(df['date'])
        return {
            'has_date': True,
            'start_date': dates.min().date(),
            'end_date': dates.max().date(),
            'total_days': (dates.max() - dates.min()).days
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结"""
        return {
            'config': {
                'strategy': self.config.strategy.value,
                'train_ratio': self.config.train_ratio,
                'lookback_window': self.config.lookback_window,
                'prediction_horizon': self.config.prediction_horizon,
            },
            'data_info': self.data_info if self.is_prepared else {},
            'trained_models': list(self.trained_models.keys()),
            'training_results': {
                name: {
                    'prediction_target': result.get('prediction_target'),
                    'training_time': result.get('training_time'),
                    'validation_metrics': result.get('validation_metrics', {})
                }
                for name, result in self.training_results.items()
            }
        }


def create_default_trainer(feature_pipeline: FeaturePipeline = None) -> UnifiedModelTrainer:
    """
    创建默认配置的模型训练器
    
    Args:
        feature_pipeline: 特征工程流水线
        
    Returns:
        UnifiedModelTrainer: 训练器实例
    """
    config = TrainingConfig(
        train_ratio=0.7,
        validation_ratio=0.15, 
        test_ratio=0.15,
        lookback_window=60,
        prediction_horizon=1,
        strategy=TrainingStrategy.TIME_SERIES_CV,
        cv_folds=5,
        enable_hyperopt=True
    )
    
    return UnifiedModelTrainer(config=config, feature_pipeline=feature_pipeline)



