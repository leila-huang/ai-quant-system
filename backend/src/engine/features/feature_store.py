"""
特征存储管理

提供特征数据的高效存储、缓存和检索功能，
支持增量更新和分布式特征计算结果存储。
"""

import os
import time
import hashlib
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from joblib import dump, load

from backend.src.models.basic_models import StockData
from backend.src.storage.parquet_engine import ParquetStorage
from backend.src.engine.utils import CacheManager, ValidationHelper

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """特征元数据"""
    feature_version: str
    created_at: datetime
    symbols: List[str]
    start_date: date
    end_date: date
    feature_count: int
    row_count: int
    pipeline_config: Dict[str, Any]
    checksum: str


class FeatureStore:
    """
    特征存储管理器
    
    提供特征数据的存储、缓存、检索和版本管理功能。
    支持增量更新和高效的特征查询。
    """
    
    def __init__(self, 
                 storage_path: str = "data/features",
                 cache_size: int = 100):
        """
        初始化特征存储管理器
        
        Args:
            storage_path: 特征存储根路径
            cache_size: 内存缓存大小
        """
        self.storage_path = Path(storage_path)
        self.cache_size = cache_size
        
        # 创建存储目录结构
        self.raw_features_path = self.storage_path / "raw_features"
        self.processed_features_path = self.storage_path / "processed_features"
        self.metadata_path = self.storage_path / "metadata"
        self.models_path = self.storage_path / "models"
        
        for path in [self.raw_features_path, self.processed_features_path, 
                     self.metadata_path, self.models_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # 不使用ParquetStorage，直接操作parquet文件
        # 特征数据与股票数据结构不同，直接保存DataFrame更合适
        pass
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager(max_size=cache_size)
        
        logger.info(f"特征存储管理器初始化完成，存储路径: {storage_path}")
    
    def store_raw_features(self, 
                          features_df: pd.DataFrame,
                          feature_version: str = None) -> str:
        """
        存储原始特征数据
        
        Args:
            features_df: 特征数据DataFrame
            feature_version: 特征版本号
            
        Returns:
            str: 存储的特征版本号
        """
        if features_df.empty:
            raise ValueError("特征数据不能为空")
        
        # 生成版本号
        if feature_version is None:
            feature_version = self._generate_feature_version(features_df)
        
        logger.info(f"存储原始特征数据，版本: {feature_version}，维度: {features_df.shape}")
        
        try:
            # 按股票代码分组存储
            if 'symbol' in features_df.columns:
                grouped = features_df.groupby('symbol')
                for symbol, group_df in grouped:
                    # 直接存储为parquet文件
                    file_path = self.raw_features_path / f"{symbol}_{feature_version}.parquet"
                    group_df.to_parquet(file_path, compression='snappy', index=False)
            else:
                # 如果没有股票代码列，则存储为单个文件
                file_path = self.raw_features_path / f"features_{feature_version}.parquet"
                features_df.to_parquet(file_path, compression='snappy', index=False)
            
            # 保存元数据
            metadata = self._create_metadata(features_df, feature_version, "raw")
            self._save_metadata(metadata, feature_version, "raw")
            
            logger.info(f"原始特征数据存储完成，版本: {feature_version}")
            return feature_version
            
        except Exception as e:
            logger.error(f"存储原始特征数据失败: {e}")
            raise
    
    def store_processed_features(self, 
                               features_df: pd.DataFrame,
                               pipeline_config: Dict[str, Any],
                               feature_version: str = None) -> str:
        """
        存储处理后的特征数据
        
        Args:
            features_df: 处理后的特征数据
            pipeline_config: 流水线配置
            feature_version: 特征版本号
            
        Returns:
            str: 存储的特征版本号
        """
        if features_df.empty:
            raise ValueError("特征数据不能为空")
        
        # 生成版本号
        if feature_version is None:
            feature_version = self._generate_feature_version(features_df, pipeline_config)
        
        logger.info(f"存储处理后特征数据，版本: {feature_version}，维度: {features_df.shape}")
        
        try:
            # 按股票代码分组存储
            if 'symbol' in features_df.columns:
                grouped = features_df.groupby('symbol')
                for symbol, group_df in grouped:
                    # 直接存储为parquet文件
                    file_path = self.processed_features_path / f"{symbol}_{feature_version}.parquet"
                    group_df.to_parquet(file_path, compression='snappy', index=False)
            else:
                # 如果没有股票代码列，则存储为单个文件
                file_path = self.processed_features_path / f"features_{feature_version}.parquet"
                features_df.to_parquet(file_path, compression='snappy', index=False)
            
            # 保存元数据
            metadata = self._create_metadata(features_df, feature_version, "processed", pipeline_config)
            self._save_metadata(metadata, feature_version, "processed")
            
            # 更新缓存
            cache_key = f"processed_features_{feature_version}"
            self.cache_manager.set(cache_key, features_df)
            
            logger.info(f"处理后特征数据存储完成，版本: {feature_version}")
            return feature_version
            
        except Exception as e:
            logger.error(f"存储处理后特征数据失败: {e}")
            raise
    
    def load_raw_features(self, 
                         symbols: List[str] = None,
                         start_date: date = None,
                         end_date: date = None,
                         feature_version: str = None) -> pd.DataFrame:
        """
        加载原始特征数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            feature_version: 特征版本号
            
        Returns:
            pd.DataFrame: 原始特征数据
        """
        logger.info(f"加载原始特征数据，股票: {symbols}, 日期: {start_date}-{end_date}")
        
        try:
            if symbols:
                # 按股票代码加载
                dfs = []
                for symbol in symbols:
                    # 查找该股票的特征文件
                    pattern = f"{symbol}_*.parquet"
                    matching_files = list(self.raw_features_path.glob(pattern))
                    
                    for file_path in matching_files:
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            # 按日期过滤
                            if 'date' in df.columns and (start_date or end_date):
                                df['date'] = pd.to_datetime(df['date'])
                                if start_date:
                                    df = df[df['date'].dt.date >= start_date]
                                if end_date:
                                    df = df[df['date'].dt.date <= end_date]
                            dfs.append(df)
                
                if dfs:
                    result_df = pd.concat(dfs, ignore_index=True)
                    logger.info(f"成功加载原始特征数据，维度: {result_df.shape}")
                    return result_df
                else:
                    logger.warning("未找到匹配的原始特征数据")
                    return pd.DataFrame()
            else:
                # 加载所有数据（需要指定版本号）
                if feature_version:
                    file_path = self.raw_features_path / f"features_{feature_version}.parquet"
                    if file_path.exists():
                        result_df = pd.read_parquet(file_path)
                        logger.info(f"成功加载原始特征数据，维度: {result_df.shape}")
                        return result_df
                
                logger.warning("未找到匹配的原始特征数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"加载原始特征数据失败: {e}")
            return pd.DataFrame()
    
    def load_processed_features(self, 
                              symbols: List[str] = None,
                              start_date: date = None,
                              end_date: date = None,
                              feature_version: str = None) -> pd.DataFrame:
        """
        加载处理后的特征数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            feature_version: 特征版本号
            
        Returns:
            pd.DataFrame: 处理后的特征数据
        """
        # 检查缓存
        if feature_version:
            cache_key = f"processed_features_{feature_version}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.info("从缓存中加载处理后特征数据")
                return self._filter_features(cached_data, symbols, start_date, end_date)
        
        logger.info(f"从存储加载处理后特征数据，股票: {symbols}, 日期: {start_date}-{end_date}")
        
        try:
            if symbols:
                # 按股票代码加载
                dfs = []
                for symbol in symbols:
                    # 查找该股票的特征文件
                    pattern = f"{symbol}_*.parquet"
                    matching_files = list(self.processed_features_path.glob(pattern))
                    
                    for file_path in matching_files:
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            # 按日期过滤
                            if 'date' in df.columns and (start_date or end_date):
                                df['date'] = pd.to_datetime(df['date'])
                                if start_date:
                                    df = df[df['date'].dt.date >= start_date]
                                if end_date:
                                    df = df[df['date'].dt.date <= end_date]
                            dfs.append(df)
                
                if dfs:
                    result_df = pd.concat(dfs, ignore_index=True)
                    logger.info(f"成功加载处理后特征数据，维度: {result_df.shape}")
                    return result_df
                else:
                    logger.warning("未找到匹配的处理后特征数据")
                    return pd.DataFrame()
            else:
                # 加载所有数据（需要指定版本号）
                if feature_version:
                    file_path = self.processed_features_path / f"features_{feature_version}.parquet"
                    if file_path.exists():
                        result_df = pd.read_parquet(file_path)
                        logger.info(f"成功加载处理后特征数据，维度: {result_df.shape}")
                        return result_df
                
                logger.warning("未找到匹配的处理后特征数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"加载处理后特征数据失败: {e}")
            return pd.DataFrame()
    
    def save_pipeline_model(self, 
                           model_obj: Any,
                           model_name: str,
                           model_version: str = None) -> str:
        """
        保存流水线模型对象（如标准化器、特征选择器）
        
        Args:
            model_obj: 模型对象
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            str: 保存的模型路径
        """
        if model_version is None:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_file = f"{model_name}_{model_version}.joblib"
        model_path = self.models_path / model_file
        
        try:
            dump(model_obj, model_path)
            logger.info(f"模型已保存: {model_path}")
            return str(model_path)
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
    
    def load_pipeline_model(self, 
                          model_name: str,
                          model_version: str = None) -> Any:
        """
        加载流水线模型对象
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            Any: 模型对象
        """
        try:
            if model_version:
                model_file = f"{model_name}_{model_version}.joblib"
                model_path = self.models_path / model_file
            else:
                # 查找最新版本
                model_pattern = f"{model_name}_*.joblib"
                model_files = list(self.models_path.glob(model_pattern))
                if not model_files:
                    raise FileNotFoundError(f"未找到模型: {model_name}")
                
                # 按文件名排序，取最新的
                model_path = sorted(model_files)[-1]
            
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            model_obj = load(model_path)
            logger.info(f"模型已加载: {model_path}")
            return model_obj
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def list_feature_versions(self, feature_type: str = "all") -> List[str]:
        """
        列出可用的特征版本
        
        Args:
            feature_type: 特征类型 ("raw", "processed", "all")
            
        Returns:
            List[str]: 特征版本列表
        """
        versions = []
        
        try:
            if feature_type in ["raw", "all"]:
                raw_metadata_files = list(self.metadata_path.glob("raw_*.json"))
                for file in raw_metadata_files:
                    version = file.stem.replace("raw_", "")
                    versions.append(f"raw_{version}")
            
            if feature_type in ["processed", "all"]:
                processed_metadata_files = list(self.metadata_path.glob("processed_*.json"))
                for file in processed_metadata_files:
                    version = file.stem.replace("processed_", "")
                    versions.append(f"processed_{version}")
            
            return sorted(versions)
            
        except Exception as e:
            logger.error(f"列出特征版本失败: {e}")
            return []
    
    def get_feature_metadata(self, 
                           feature_version: str,
                           feature_type: str = "processed") -> Optional[FeatureMetadata]:
        """
        获取特征元数据
        
        Args:
            feature_version: 特征版本号
            feature_type: 特征类型 ("raw", "processed")
            
        Returns:
            Optional[FeatureMetadata]: 特征元数据
        """
        try:
            metadata_file = self.metadata_path / f"{feature_type}_{feature_version}.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                
                # 转换日期字符串
                metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                metadata_dict['start_date'] = date.fromisoformat(metadata_dict['start_date'])
                metadata_dict['end_date'] = date.fromisoformat(metadata_dict['end_date'])
                
                return FeatureMetadata(**metadata_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"获取特征元数据失败: {e}")
            return None
    
    def clean_old_features(self, keep_versions: int = 5):
        """
        清理旧的特征数据，保留最新的几个版本
        
        Args:
            keep_versions: 保留的版本数量
        """
        logger.info(f"开始清理旧特征数据，保留最新 {keep_versions} 个版本")
        
        try:
            # 清理原始特征
            raw_versions = self.list_feature_versions("raw")
            if len(raw_versions) > keep_versions:
                versions_to_delete = raw_versions[:-keep_versions]
                for version in versions_to_delete:
                    self._delete_feature_version(version.replace("raw_", ""), "raw")
            
            # 清理处理后特征
            processed_versions = self.list_feature_versions("processed")
            if len(processed_versions) > keep_versions:
                versions_to_delete = processed_versions[:-keep_versions]
                for version in versions_to_delete:
                    self._delete_feature_version(version.replace("processed_", ""), "processed")
            
            logger.info("旧特征数据清理完成")
            
        except Exception as e:
            logger.error(f"清理旧特征数据失败: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        info = {
            'storage_path': str(self.storage_path),
            'raw_features_count': len(list(self.raw_features_path.glob("*.parquet"))),
            'processed_features_count': len(list(self.processed_features_path.glob("*.parquet"))),
            'models_count': len(list(self.models_path.glob("*.joblib"))),
            'cache_size': self.cache_manager.size(),
            'available_versions': {
                'raw': self.list_feature_versions("raw"),
                'processed': self.list_feature_versions("processed")
            }
        }
        
        # 计算存储大小
        def get_dir_size(path):
            total = 0
            for file in Path(path).rglob('*'):
                if file.is_file():
                    total += file.stat().st_size
            return total
        
        info['storage_size'] = {
            'raw_features_mb': get_dir_size(self.raw_features_path) / 1024 / 1024,
            'processed_features_mb': get_dir_size(self.processed_features_path) / 1024 / 1024,
            'models_mb': get_dir_size(self.models_path) / 1024 / 1024,
            'total_mb': get_dir_size(self.storage_path) / 1024 / 1024
        }
        
        return info
    
    def _generate_feature_version(self, 
                                features_df: pd.DataFrame,
                                pipeline_config: Dict[str, Any] = None) -> str:
        """生成特征版本号"""
        # 基于数据内容和配置生成哈希
        content = f"{features_df.shape}_{features_df.columns.tolist()}"
        if pipeline_config:
            content += f"_{str(pipeline_config)}"
        
        hash_obj = hashlib.md5(content.encode())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{timestamp}_{hash_obj.hexdigest()[:8]}"
    
    def _create_metadata(self, 
                        features_df: pd.DataFrame,
                        feature_version: str,
                        feature_type: str,
                        pipeline_config: Dict[str, Any] = None) -> FeatureMetadata:
        """创建特征元数据"""
        symbols = []
        if 'symbol' in features_df.columns:
            symbols = features_df['symbol'].unique().tolist()
        
        start_date = date.today()
        end_date = date.today()
        if 'date' in features_df.columns:
            dates = pd.to_datetime(features_df['date'])
            start_date = dates.min().date()
            end_date = dates.max().date()
        
        # 生成数据校验和
        checksum = hashlib.md5(str(features_df.values.tobytes()).encode()).hexdigest()
        
        return FeatureMetadata(
            feature_version=feature_version,
            created_at=datetime.now(),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            feature_count=len(features_df.columns),
            row_count=len(features_df),
            pipeline_config=pipeline_config or {},
            checksum=checksum
        )
    
    def _save_metadata(self, 
                      metadata: FeatureMetadata,
                      feature_version: str,
                      feature_type: str):
        """保存特征元数据"""
        import json
        
        metadata_dict = asdict(metadata)
        # 转换日期为字符串
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        metadata_dict['start_date'] = metadata.start_date.isoformat()
        metadata_dict['end_date'] = metadata.end_date.isoformat()
        
        metadata_file = self.metadata_path / f"{feature_type}_{feature_version}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    
    def _filter_features(self, 
                        features_df: pd.DataFrame,
                        symbols: List[str] = None,
                        start_date: date = None,
                        end_date: date = None) -> pd.DataFrame:
        """过滤特征数据"""
        if features_df.empty:
            return features_df
        
        filtered_df = features_df.copy()
        
        # 按股票代码过滤
        if symbols and 'symbol' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['symbol'].isin(symbols)]
        
        # 按日期过滤
        if 'date' in filtered_df.columns:
            if start_date:
                filtered_df = filtered_df[pd.to_datetime(filtered_df['date']).dt.date >= start_date]
            if end_date:
                filtered_df = filtered_df[pd.to_datetime(filtered_df['date']).dt.date <= end_date]
        
        return filtered_df
    
    def _delete_feature_version(self, feature_version: str, feature_type: str):
        """删除指定版本的特征数据"""
        try:
            # 删除数据文件
            if feature_type == "raw":
                storage = self.raw_storage
            else:
                storage = self.processed_storage
            
            # 删除元数据文件
            metadata_file = self.metadata_path / f"{feature_type}_{feature_version}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            logger.info(f"已删除特征版本: {feature_type}_{feature_version}")
            
        except Exception as e:
            logger.error(f"删除特征版本失败: {e}")


def create_default_feature_store(storage_path: str = "data/features") -> FeatureStore:
    """
    创建默认配置的特征存储管理器
    
    Args:
        storage_path: 存储路径
        
    Returns:
        FeatureStore: 特征存储管理器实例
    """
    return FeatureStore(storage_path=storage_path, cache_size=100)
