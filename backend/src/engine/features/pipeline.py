"""
特征工程数据流水线

提供从原始股票数据到机器学习特征的完整转换流水线，
支持批量处理、并行计算、特征缓存和增量更新。
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from joblib import Parallel, delayed

from backend.src.models.basic_models import StockData, StockDailyBar
from backend.src.engine.features import FeaturePipeline as BaseFeaturePipeline
from backend.src.engine.features.indicators import TechnicalIndicators
from backend.src.storage.parquet_engine import ParquetStorage
from backend.src.engine.utils import PerformanceMonitor, DataConverter, ValidationHelper

logger = logging.getLogger(__name__)


class ScalerType(Enum):
    """数据标准化类型枚举"""
    STANDARD = "standard"          # 标准化 (Z-score)
    MINMAX = "minmax"             # 最小-最大标准化
    ROBUST = "robust"             # 鲁棒标准化
    NONE = "none"                 # 不标准化


class FeatureSelectionMethod(Enum):
    """特征选择方法枚举"""
    K_BEST_F = "k_best_f"                    # F统计量选择K个最佳特征
    MUTUAL_INFO = "mutual_info"              # 互信息特征选择
    CORRELATION = "correlation"              # 相关性特征选择
    VARIANCE = "variance"                    # 方差特征选择
    NONE = "none"                           # 不进行特征选择


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 技术指标配置
    indicators: List[str] = None
    ma_windows: List[int] = None
    rsi_windows: List[int] = None
    
    # 特征转换配置
    scaler_type: ScalerType = ScalerType.STANDARD
    feature_selection: FeatureSelectionMethod = FeatureSelectionMethod.K_BEST_F
    n_features: int = 20
    
    # 性能配置
    max_workers: int = 4
    batch_size: int = 10
    cache_features: bool = True
    
    # 数据质量配置
    min_data_points: int = 30
    max_missing_ratio: float = 0.1
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = ["MA", "EMA", "RSI", "MACD", "BOLL", "KDJ", "WILLIAMS", "VOLUME_MA"]
        if self.ma_windows is None:
            self.ma_windows = [5, 10, 20, 30, 60]
        if self.rsi_windows is None:
            self.rsi_windows = [14, 6]


class FeaturePipeline(BaseFeaturePipeline):
    """
    特征工程数据流水线
    
    提供从原始股票数据到机器学习特征的完整转换过程，
    包括技术指标计算、特征变换、特征选择、数据标准化等。
    """
    
    def __init__(self, 
                 config: PipelineConfig = None,
                 storage_path: str = "data/features"):
        """
        初始化特征工程流水线
        
        Args:
            config: 流水线配置
            storage_path: 特征存储路径
        """
        self.config = config or PipelineConfig()
        self.storage_path = storage_path
        
        # 初始化组件
        self.indicator_calculator = TechnicalIndicators()
        self.feature_storage = ParquetStorage(
            base_path=os.path.join(storage_path, "raw_features"),
            max_workers=self.config.max_workers
        )
        self.processed_storage = ParquetStorage(
            base_path=os.path.join(storage_path, "processed_features"),
            max_workers=self.config.max_workers
        )
        
        # 流水线组件
        self._scaler = None
        self._feature_selector = None
        self._selected_features = None
        self._is_fitted = False
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"特征工程流水线初始化完成，存储路径: {storage_path}")
    
    def fit(self, data: Union[List[StockData], Dict[str, StockData], StockData]) -> 'FeaturePipeline':
        """
        训练特征流水线
        
        Args:
            data: 训练数据
            
        Returns:
            self: 返回自身以支持链式调用
        """
        logger.info("开始训练特征工程流水线...")
        start_time = time.time()
        
        try:
            # 数据预处理
            stock_data_list = self._prepare_input_data(data)
            
            if not stock_data_list:
                raise ValueError("训练数据为空")
            
            logger.info(f"训练数据包含 {len(stock_data_list)} 只股票")
            
            # 计算原始特征
            raw_features = self._compute_raw_features_batch(stock_data_list)
            
            if raw_features.empty:
                raise ValueError("无法计算特征，请检查输入数据")
            
            # 合并所有特征数据用于训练
            combined_features = self._combine_features_for_training(raw_features)
            
            # 训练标准化器
            self._fit_scaler(combined_features)
            
            # 训练特征选择器
            self._fit_feature_selector(combined_features)
            
            self._is_fitted = True
            
            end_time = time.time()
            logger.info(f"特征流水线训练完成，耗时: {end_time - start_time:.2f}秒")
            
            return self
            
        except Exception as e:
            logger.error(f"特征流水线训练失败: {e}")
            raise
    
    def transform(self, data: Union[List[StockData], Dict[str, StockData], StockData]) -> pd.DataFrame:
        """
        应用特征转换
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 转换后的特征数据
        """
        if not self._is_fitted:
            raise ValueError("流水线尚未训练，请先调用fit()方法")
        
        logger.info("开始特征转换...")
        start_time = time.time()
        
        try:
            # 数据预处理
            stock_data_list = self._prepare_input_data(data)
            
            if not stock_data_list:
                raise ValueError("输入数据为空")
            
            # 计算原始特征
            raw_features = self._compute_raw_features_batch(stock_data_list)
            
            if raw_features.empty:
                logger.warning("无法计算特征，返回空DataFrame")
                return pd.DataFrame()
            
            # 应用特征转换
            processed_features = self._apply_feature_transformations(raw_features)
            
            end_time = time.time()
            logger.info(f"特征转换完成，耗时: {end_time - start_time:.2f}秒，输出维度: {processed_features.shape}")
            
            return processed_features
            
        except Exception as e:
            logger.error(f"特征转换失败: {e}")
            raise
    
    def compute_features_for_symbols(self, 
                                   symbols: List[str],
                                   start_date: date,
                                   end_date: date,
                                   data_source: Callable = None) -> pd.DataFrame:
        """
        为指定股票列表计算特征
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_source: 数据源函数，用于获取股票数据
            
        Returns:
            pd.DataFrame: 特征数据
        """
        logger.info(f"开始为 {len(symbols)} 只股票计算特征: {start_date} 到 {end_date}")
        
        if data_source is None:
            raise ValueError("需要提供数据源函数")
        
        # 并行获取股票数据
        stock_data_list = self._fetch_stock_data_parallel(symbols, start_date, end_date, data_source)
        
        # 计算特征
        if self._is_fitted:
            return self.transform(stock_data_list)
        else:
            # 如果未训练，则进行fit_transform
            return self.fit_transform(stock_data_list)
    
    def get_feature_importance(self, feature_data: pd.DataFrame, 
                             target: pd.Series) -> Dict[str, float]:
        """
        计算特征重要性
        
        Args:
            feature_data: 特征数据
            target: 目标变量
            
        Returns:
            Dict[str, float]: 特征重要性字典
        """
        logger.info("开始计算特征重要性...")
        
        try:
            # 移除非数值列
            numeric_features = feature_data.select_dtypes(include=[np.number])
            
            if numeric_features.empty:
                logger.warning("没有数值特征可用于重要性计算")
                return {}
            
            # 处理缺失值
            feature_clean = numeric_features.fillna(numeric_features.median())
            target_clean = target.fillna(target.median())
            
            # 计算F统计量
            f_scores, _ = f_regression(feature_clean, target_clean)
            
            # 计算互信息
            mi_scores = mutual_info_regression(feature_clean, target_clean, random_state=42)
            
            # 合并重要性分数
            importance = {}
            for i, feature in enumerate(feature_clean.columns):
                importance[feature] = {
                    'f_score': f_scores[i],
                    'mutual_info': mi_scores[i],
                    'combined': (f_scores[i] + mi_scores[i]) / 2
                }
            
            # 按组合分数排序
            sorted_importance = dict(sorted(
                importance.items(), 
                key=lambda x: x[1]['combined'], 
                reverse=True
            ))
            
            logger.info(f"特征重要性计算完成，共 {len(sorted_importance)} 个特征")
            return sorted_importance
            
        except Exception as e:
            logger.error(f"特征重要性计算失败: {e}")
            return {}
    
    def _prepare_input_data(self, data: Union[List[StockData], Dict[str, StockData], StockData]) -> List[StockData]:
        """准备输入数据"""
        if isinstance(data, StockData):
            return [data]
        elif isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.values())
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
    
    def _compute_raw_features_batch(self, stock_data_list: List[StockData]) -> pd.DataFrame:
        """批量计算原始技术指标特征"""
        start_time = time.time()
        logger.info(f"开始批量计算 {len(stock_data_list)} 只股票的技术指标...")
        
        def compute_single_stock_features(stock_data):
            try:
                # 检查数据质量
                if not stock_data.bars or len(stock_data.bars) < self.config.min_data_points:
                    logger.warning(f"股票 {stock_data.symbol} 数据不足，跳过")
                    return None
                
                # 计算技术指标
                indicators_df = self.indicator_calculator.calculate(
                    stock_data,
                    indicators=self.config.indicators,
                    ma_windows=self.config.ma_windows,
                    rsi_windows=self.config.rsi_windows
                )
                
                if indicators_df.empty:
                    logger.warning(f"股票 {stock_data.symbol} 技术指标计算失败")
                    return None
                
                # 添加基础信息
                indicators_df['symbol'] = stock_data.symbol
                indicators_df['name'] = stock_data.name or ''
                
                return indicators_df
                
            except Exception as e:
                logger.error(f"计算股票 {stock_data.symbol} 特征失败: {e}")
                return None
        
        # 并行计算
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_symbol = {
                executor.submit(compute_single_stock_features, stock_data): stock_data.symbol 
                for stock_data in stock_data_list
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"处理股票 {symbol} 时发生异常: {e}")
        
        if results:
            combined_df = pd.concat(results, ignore_index=True)
            end_time = time.time()
            self.performance_monitor._record_metric("compute_raw_features_batch", end_time - start_time)
            logger.info(f"成功计算 {len(results)} 只股票的特征，总维度: {combined_df.shape}，耗时: {end_time - start_time:.3f}秒")
            return combined_df
        else:
            end_time = time.time()
            self.performance_monitor._record_metric("compute_raw_features_batch", end_time - start_time)
            logger.warning("没有成功计算任何股票的特征")
            return pd.DataFrame()
    
    def _combine_features_for_training(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """合并特征数据用于训练标准化器和特征选择器"""
        if features_df.empty:
            return pd.DataFrame()
        
        # 选择数值特征用于训练
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除一些不需要标准化的列
        exclude_cols = ['symbol', 'name', 'date']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        return features_df[numeric_cols].copy()
    
    def _fit_scaler(self, features_df: pd.DataFrame):
        """训练数据标准化器"""
        if self.config.scaler_type == ScalerType.NONE or features_df.empty:
            self._scaler = None
            return
        
        logger.info(f"训练数据标准化器: {self.config.scaler_type.value}")
        
        # 处理缺失值
        features_clean = features_df.fillna(features_df.median())
        
        if self.config.scaler_type == ScalerType.STANDARD:
            self._scaler = StandardScaler()
        elif self.config.scaler_type == ScalerType.MINMAX:
            self._scaler = MinMaxScaler()
        elif self.config.scaler_type == ScalerType.ROBUST:
            self._scaler = RobustScaler()
        
        self._scaler.fit(features_clean)
        logger.info("数据标准化器训练完成")
    
    def _fit_feature_selector(self, features_df: pd.DataFrame):
        """训练特征选择器"""
        if self.config.feature_selection == FeatureSelectionMethod.NONE or features_df.empty:
            self._feature_selector = None
            self._selected_features = features_df.columns.tolist()
            return
        
        logger.info(f"训练特征选择器: {self.config.feature_selection.value}")
        
        # 创建虚拟目标变量（使用收盘价的未来收益率）
        if 'close' in features_df.columns:
            target = features_df['close'].pct_change(1).shift(-1).fillna(0)
        else:
            logger.warning("没有找到收盘价列，跳过特征选择")
            self._feature_selector = None
            self._selected_features = features_df.columns.tolist()
            return
        
        # 处理缺失值
        features_clean = features_df.fillna(features_df.median())
        target_clean = target.fillna(0)
        
        n_features = min(self.config.n_features, len(features_clean.columns))
        
        if self.config.feature_selection == FeatureSelectionMethod.K_BEST_F:
            self._feature_selector = SelectKBest(f_regression, k=n_features)
        elif self.config.feature_selection == FeatureSelectionMethod.MUTUAL_INFO:
            self._feature_selector = SelectKBest(mutual_info_regression, k=n_features)
        
        self._feature_selector.fit(features_clean, target_clean)
        
        # 获取选择的特征名称
        selected_mask = self._feature_selector.get_support()
        self._selected_features = features_clean.columns[selected_mask].tolist()
        
        logger.info(f"特征选择完成，从 {len(features_clean.columns)} 个特征中选择了 {len(self._selected_features)} 个")
    
    def _apply_feature_transformations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """应用特征转换"""
        if features_df.empty:
            return features_df
        
        result_df = features_df.copy()
        
        # 获取数值特征
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['symbol', 'name', 'date']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 应用标准化
        if self._scaler is not None and numeric_cols:
            logger.info("应用数据标准化...")
            
            # 处理缺失值
            features_clean = result_df[numeric_cols].fillna(result_df[numeric_cols].median())
            
            # 应用标准化
            scaled_features = self._scaler.transform(features_clean)
            result_df[numeric_cols] = scaled_features
        
        # 应用特征选择
        if self._feature_selector is not None and self._selected_features:
            logger.info(f"应用特征选择，保留 {len(self._selected_features)} 个特征...")
            
            # 保留非数值列和选中的特征
            non_numeric_cols = [col for col in result_df.columns if col not in numeric_cols]
            selected_cols = non_numeric_cols + self._selected_features
            result_df = result_df[selected_cols]
        
        return result_df
    
    def _fetch_stock_data_parallel(self, 
                                 symbols: List[str],
                                 start_date: date,
                                 end_date: date,
                                 data_source: Callable) -> List[StockData]:
        """并行获取股票数据"""
        logger.info(f"并行获取 {len(symbols)} 只股票数据...")
        
        def fetch_single_stock(symbol):
            try:
                return data_source(symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"获取股票 {symbol} 数据失败: {e}")
                return None
        
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_symbol = {
                executor.submit(fetch_single_stock, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"处理股票 {symbol} 时发生异常: {e}")
        
        logger.info(f"成功获取 {len(results)} 只股票数据")
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """获取流水线信息"""
        return {
            'config': {
                'indicators': self.config.indicators,
                'scaler_type': self.config.scaler_type.value,
                'feature_selection': self.config.feature_selection.value,
                'n_features': self.config.n_features,
                'max_workers': self.config.max_workers,
            },
            'status': {
                'is_fitted': self._is_fitted,
                'selected_features_count': len(self._selected_features) if self._selected_features else 0,
                'has_scaler': self._scaler is not None,
                'has_feature_selector': self._feature_selector is not None,
            },
            'selected_features': self._selected_features,
            'performance': self.performance_monitor.get_stats('compute_raw_features_batch')
        }


def create_default_pipeline(storage_path: str = "data/features") -> FeaturePipeline:
    """
    创建默认配置的特征工程流水线
    
    Args:
        storage_path: 存储路径
        
    Returns:
        FeaturePipeline: 配置好的流水线实例
    """
    config = PipelineConfig(
        indicators=["MA", "EMA", "RSI", "MACD", "BOLL", "KDJ", "WILLIAMS", "VOLUME_MA"],
        ma_windows=[5, 10, 20, 30, 60],
        rsi_windows=[14, 6],
        scaler_type=ScalerType.STANDARD,
        feature_selection=FeatureSelectionMethod.K_BEST_F,
        n_features=30,
        max_workers=4,
        batch_size=10,
        cache_features=True
    )
    
    return FeaturePipeline(config, storage_path)
