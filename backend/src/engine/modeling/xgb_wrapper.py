"""
XGBoost模型封装

提供完整的XGBoost建模框架，包括模型训练、预测、评估、
超参数优化，专门针对金融时间序列数据进行优化。
"""

import os
import warnings
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

from backend.src.engine.modeling import (
    ModelTrainer, ModelEvaluator, PredictionTarget, SUPPORTED_METRICS
)
from backend.src.engine.utils import PerformanceMonitor, ValidationHelper

# 禁用xgboost警告
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

logger = logging.getLogger(__name__)


class XGBoostModelFramework(ModelTrainer):
    """
    XGBoost建模框架
    
    针对金融时间序列数据优化的XGBoost建模解决方案，
    支持多种预测目标和完整的模型生命周期管理。
    """
    
    def __init__(self, 
                 prediction_target: PredictionTarget = PredictionTarget.RETURN,
                 model_params: Optional[Dict[str, Any]] = None,
                 enable_gpu: bool = False):
        """
        初始化XGBoost建模框架
        
        Args:
            prediction_target: 预测目标类型
            model_params: 模型参数
            enable_gpu: 是否启用GPU加速
        """
        self.prediction_target = prediction_target
        self.enable_gpu = enable_gpu
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        
        # 设置默认模型参数
        self.model_params = self._get_default_params()
        if model_params:
            self.model_params.update(model_params)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 训练历史
        self.training_history = {
            'train_scores': [],
            'val_scores': [],
            'feature_importance': {},
            'hyperparams_history': []
        }
        
        logger.info(f"XGBoost建模框架初始化完成，预测目标: {prediction_target.value}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认模型参数"""
        base_params = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 1,
            'colsample_bynode': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # 根据预测目标调整参数
        if self.prediction_target == PredictionTarget.DIRECTION:
            base_params['objective'] = 'binary:logistic'
            base_params['eval_metric'] = 'logloss'
        elif self.prediction_target == PredictionTarget.CLASSIFICATION:
            base_params['objective'] = 'multi:softprob'
            base_params['eval_metric'] = 'mlogloss'
        else:
            base_params['eval_metric'] = 'rmse'
        
        # GPU支持
        if self.enable_gpu:
            try:
                base_params['tree_method'] = 'gpu_hist'
                base_params['gpu_id'] = 0
            except Exception as e:
                logger.warning(f"GPU不可用，使用CPU训练: {e}")
                self.enable_gpu = False
        
        return base_params
    
    def prepare_training_data(self, 
                            features_df: pd.DataFrame,
                            target_column: str = None,
                            lookback_window: int = 30,
                            prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据，考虑金融时间序列特点
        
        Args:
            features_df: 特征数据
            target_column: 目标列名（如果为None则自动生成）
            lookback_window: 回看窗口
            prediction_horizon: 预测时间范围
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特征和目标变量
        """
        logger.info(f"准备训练数据，预测目标: {self.prediction_target.value}")
        
        if features_df.empty:
            raise ValueError("特征数据不能为空")
        
        # 确保数据按时间排序
        if 'date' in features_df.columns:
            features_df = features_df.sort_values('date').reset_index(drop=True)
        
        # 生成目标变量
        if target_column and target_column in features_df.columns:
            y = features_df[target_column].copy()
        else:
            y = self._generate_target_variable(features_df, prediction_horizon)
        
        # 准备特征数据
        X = self._prepare_features(features_df, lookback_window)
        
        # 移除包含NaN的行（由于滞后特征和目标变量生成）
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        logger.info(f"训练数据准备完成，样本数: {len(X_clean)}, 特征数: {len(X_clean.columns)}")
        
        return X_clean, y_clean
    
    def _generate_target_variable(self, 
                                features_df: pd.DataFrame, 
                                prediction_horizon: int) -> pd.Series:
        """根据预测目标生成目标变量"""
        if 'close' not in features_df.columns:
            raise ValueError("数据中缺少'close'列，无法生成目标变量")
        
        close_prices = features_df['close']
        
        if self.prediction_target == PredictionTarget.PRICE:
            # 价格预测：预测future price
            target = close_prices.shift(-prediction_horizon)
            
        elif self.prediction_target == PredictionTarget.RETURN:
            # 收益率预测：预测future return
            future_prices = close_prices.shift(-prediction_horizon)
            target = (future_prices - close_prices) / close_prices
            
        elif self.prediction_target == PredictionTarget.DIRECTION:
            # 方向预测：预测涨跌方向（二分类）
            future_prices = close_prices.shift(-prediction_horizon)
            target = (future_prices > close_prices).astype(int)
            
        elif self.prediction_target == PredictionTarget.VOLATILITY:
            # 波动率预测：预测未来波动率
            returns = close_prices.pct_change()
            target = returns.rolling(window=prediction_horizon).std().shift(-prediction_horizon)
            
        else:
            raise ValueError(f"不支持的预测目标: {self.prediction_target}")
        
        return target
    
    def _prepare_features(self, 
                         features_df: pd.DataFrame, 
                         lookback_window: int) -> pd.DataFrame:
        """准备特征数据，添加滞后特征避免数据泄露"""
        # 选择数值特征
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['date', 'symbol', 'name']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = features_df[feature_cols].copy()
        
        # 添加滞后特征（避免数据泄露）
        lagged_features = []
        for lag in [1, 2, 3, 5]:  # 1日、2日、3日、5日滞后
            if lag <= lookback_window:
                lagged_df = X.shift(lag)
                lagged_df.columns = [f'{col}_lag_{lag}' for col in lagged_df.columns]
                lagged_features.append(lagged_df)
        
        # 添加移动平均特征
        ma_features = []
        for window in [5, 10, 20]:
            if window <= lookback_window:
                ma_df = X.rolling(window=window, min_periods=1).mean()
                ma_df.columns = [f'{col}_ma_{window}' for col in ma_df.columns]
                ma_features.append(ma_df)
        
        # 合并所有特征
        all_features = [X] + lagged_features + ma_features
        X_enhanced = pd.concat(all_features, axis=1)
        
        return X_enhanced
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2, **kwargs) -> 'XGBoostModelFramework':
        """
        训练XGBoost模型
        
        Args:
            X: 特征数据
            y: 目标变量
            validation_split: 验证集比例
            **kwargs: 额外参数
            
        Returns:
            self: 返回自身以支持链式调用
        """
        logger.info("开始训练XGBoost模型...")
        start_time = datetime.now()
        
        try:
            # 数据验证
            if X.empty or y.empty:
                raise ValueError("训练数据不能为空")
            
            if len(X) != len(y):
                raise ValueError("特征和目标变量长度不匹配")
            
            # 处理分类问题的标签编码
            y_processed = self._prepare_target_variable(y)
            
            # 时间序列分割（避免数据泄露）
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y_processed.iloc[:split_idx], y_processed.iloc[split_idx:]
            
            logger.info(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
            
            # 创建XGBoost模型
            self.model = xgb.XGBRegressor(**self.model_params)
            if self.prediction_target in [PredictionTarget.DIRECTION, PredictionTarget.CLASSIFICATION]:
                self.model = xgb.XGBClassifier(**self.model_params)
            
            # 训练模型 - 适配XGBoost 3.0+ sklearn接口
            # 暂时使用基本训练，不使用早期停止以确保兼容性
            self.model.fit(X_train, y_train)
            
            # 记录训练历史
            self._record_training_history(X_train, y_train, X_val, y_val)
            
            self.is_trained = True
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            logger.info(f"XGBoost模型训练完成，耗时: {training_time:.2f}秒")
            return self
            
        except Exception as e:
            logger.error(f"XGBoost模型训练失败: {e}")
            raise
    
    def _prepare_target_variable(self, y: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """处理目标变量，特别是分类问题"""
        # 确保y是Series格式
        if isinstance(y, pd.DataFrame):
            if len(y.columns) == 1:
                y = y.iloc[:, 0]  # 取第一列
            else:
                raise ValueError("目标变量DataFrame包含多列，请指定目标列")
        
        if self.prediction_target == PredictionTarget.CLASSIFICATION:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = pd.Series(self.label_encoder.fit_transform(y), index=y.index)
            else:
                y_encoded = pd.Series(self.label_encoder.transform(y), index=y.index)
            return y_encoded
        
        return y
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测结果
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        if X.empty:
            raise ValueError("预测数据不能为空")
        
        try:
            predictions = self.model.predict(X)
            
            # 处理分类问题的标签解码
            if self.prediction_target == PredictionTarget.CLASSIFICATION and self.label_encoder:
                predictions = self.label_encoder.inverse_transform(predictions.astype(int))
            
            logger.info(f"预测完成，样本数: {len(predictions)}")
            return predictions
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率（仅适用于分类问题）
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测概率
        """
        if self.prediction_target not in [PredictionTarget.DIRECTION, PredictionTarget.CLASSIFICATION]:
            raise ValueError("只有分类问题才能预测概率")
        
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            Dict[str, float]: 特征重要性字典
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练")
        
        feature_names = self.model.feature_names_in_
        importance_scores = self.model.feature_importances_
        
        importance_dict = dict(zip(feature_names, importance_scores))
        
        # 按重要性排序
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importance
    
    def optimize_hyperparameters(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                param_grid: Optional[Dict[str, List]] = None,
                                cv_folds: int = 3,
                                scoring: str = None) -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            X: 特征数据
            y: 目标变量
            param_grid: 参数网格
            cv_folds: 交叉验证折数
            scoring: 评分标准
            
        Returns:
            Dict[str, Any]: 最优参数
        """
        logger.info("开始超参数优化...")
        
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        if scoring is None:
            scoring = self._get_default_scoring()
        
        # 准备目标变量
        y_processed = self._prepare_target_variable(y)
        
        # 创建基础模型
        base_model = xgb.XGBRegressor(**self._get_base_params())
        if self.prediction_target in [PredictionTarget.DIRECTION, PredictionTarget.CLASSIFICATION]:
            base_model = xgb.XGBClassifier(**self._get_base_params())
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y_processed)
        
        # 更新模型参数
        best_params = grid_search.best_params_
        self.model_params.update(best_params)
        
        # 记录优化历史
        self.training_history['hyperparams_history'].append({
            'timestamp': datetime.now(),
            'best_params': best_params,
            'best_score': grid_search.best_score_
        })
        
        logger.info(f"超参数优化完成，最优分数: {grid_search.best_score_:.4f}")
        logger.info(f"最优参数: {best_params}")
        
        return best_params
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """获取默认参数网格"""
        return {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'n_estimators': [50, 100, 200],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
    
    def _get_default_scoring(self) -> str:
        """获取默认评分标准"""
        if self.prediction_target == PredictionTarget.DIRECTION:
            return 'roc_auc'
        elif self.prediction_target == PredictionTarget.CLASSIFICATION:
            return 'f1_weighted'
        else:
            return 'neg_mean_squared_error'
    
    def _get_base_params(self) -> Dict[str, Any]:
        """获取基础参数（用于网格搜索）"""
        return {
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def save_model(self, path: str) -> bool:
        """
        保存模型
        
        Args:
            path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练")
        
        try:
            # 创建目录
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存模型和相关信息
            model_data = {
                'model': self.model,
                'prediction_target': self.prediction_target,
                'model_params': self.model_params,
                'label_encoder': self.label_encoder,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            
            dump(model_data, path)
            logger.info(f"模型已保存到: {path}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"模型文件不存在: {path}")
            
            model_data = load(path)
            
            self.model = model_data['model']
            self.prediction_target = model_data['prediction_target']
            self.model_params = model_data['model_params']
            self.label_encoder = model_data.get('label_encoder')
            self.training_history = model_data.get('training_history', {})
            self.is_trained = model_data['is_trained']
            
            logger.info(f"模型已从 {path} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def _record_training_history(self, X_train, y_train, X_val, y_val):
        """记录训练历史"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("记录训练历史失败: 模型尚未训练")
                return
                
            # 确保目标变量是Series格式
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.iloc[:, 0]
            if isinstance(y_val, pd.DataFrame):
                y_val = y_val.iloc[:, 0]
            
            # 训练集评分
            train_pred = self.model.predict(X_train)
            train_score = self._calculate_score(y_train, train_pred)
            
            # 验证集评分
            val_pred = self.model.predict(X_val)
            val_score = self._calculate_score(y_val, val_pred)
            
            self.training_history['train_scores'].append(train_score)
            self.training_history['val_scores'].append(val_score)
            
            # 特征重要性
            if hasattr(self.model, 'feature_importances_'):
                self.training_history['feature_importance'] = self.get_feature_importance()
            
        except Exception as e:
            logger.warning(f"记录训练历史失败: {e}")
    
    def _calculate_score(self, y_true, y_pred) -> float:
        """计算模型分数"""
        # 确保目标变量是Series或numpy数组格式
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.iloc[:, 0]
        if hasattr(y_true, 'values'):
            y_true = y_true.values
            
        if self.prediction_target == PredictionTarget.DIRECTION:
            return accuracy_score(y_true, y_pred)
        elif self.prediction_target == PredictionTarget.CLASSIFICATION:
            return f1_score(y_true, y_pred, average='weighted')
        else:
            return r2_score(y_true, y_pred)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'prediction_target': self.prediction_target.value,
            'is_trained': self.is_trained,
            'model_params': self.model_params,
            'enable_gpu': self.enable_gpu,
            'training_history': self.training_history
        }
        
        if self.is_trained and self.model:
            info.update({
                'n_features': getattr(self.model, 'n_features_in_', 'Unknown'),
                'n_estimators': getattr(self.model, 'n_estimators', 'Unknown'),
                'feature_importance_top5': dict(list(self.get_feature_importance().items())[:5])
            })
        
        return info


def create_default_xgboost_framework(prediction_target: PredictionTarget = PredictionTarget.RETURN) -> XGBoostModelFramework:
    """
    创建默认配置的XGBoost建模框架
    
    Args:
        prediction_target: 预测目标
        
    Returns:
        XGBoostModelFramework: 框架实例
    """
    return XGBoostModelFramework(prediction_target=prediction_target)
