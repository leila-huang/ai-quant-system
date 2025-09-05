"""
数据压缩工具

提供多种压缩算法的统一接口，用于优化存储空间和I/O性能。
"""

import gzip
import bz2
import lzma
import snappy
from typing import Optional, Dict, Any
from enum import Enum
from loguru import logger


class CompressionType(Enum):
    """压缩类型枚举"""
    NONE = "none"
    GZIP = "gzip" 
    BZIP2 = "bzip2"
    LZMA = "lzma"
    SNAPPY = "snappy"


class CompressionUtils:
    """压缩工具类"""
    
    @staticmethod
    def compress_data(data: bytes, 
                     compression_type: CompressionType = CompressionType.SNAPPY,
                     compression_level: Optional[int] = None) -> bytes:
        """
        压缩数据
        
        Args:
            data: 原始数据
            compression_type: 压缩算法
            compression_level: 压缩级别 (仅部分算法支持)
        
        Returns:
            压缩后的数据
        """
        try:
            if compression_type == CompressionType.NONE:
                return data
            elif compression_type == CompressionType.GZIP:
                level = compression_level or 6
                return gzip.compress(data, compresslevel=level)
            elif compression_type == CompressionType.BZIP2:
                level = compression_level or 9
                return bz2.compress(data, compresslevel=level)
            elif compression_type == CompressionType.LZMA:
                preset = compression_level or 6
                return lzma.compress(data, preset=preset)
            elif compression_type == CompressionType.SNAPPY:
                try:
                    return snappy.compress(data)
                except ImportError:
                    logger.warning("snappy not available, falling back to gzip")
                    return gzip.compress(data)
            else:
                logger.error(f"Unsupported compression type: {compression_type}")
                return data
                
        except Exception as e:
            logger.error(f"Error compressing data with {compression_type}: {e}")
            return data
    
    @staticmethod
    def decompress_data(compressed_data: bytes,
                       compression_type: CompressionType = CompressionType.SNAPPY) -> bytes:
        """
        解压数据
        
        Args:
            compressed_data: 压缩数据
            compression_type: 压缩算法
        
        Returns:
            解压后的数据
        """
        try:
            if compression_type == CompressionType.NONE:
                return compressed_data
            elif compression_type == CompressionType.GZIP:
                return gzip.decompress(compressed_data)
            elif compression_type == CompressionType.BZIP2:
                return bz2.decompress(compressed_data)
            elif compression_type == CompressionType.LZMA:
                return lzma.decompress(compressed_data)
            elif compression_type == CompressionType.SNAPPY:
                try:
                    return snappy.decompress(compressed_data)
                except ImportError:
                    logger.warning("snappy not available, assuming gzip")
                    return gzip.decompress(compressed_data)
            else:
                logger.error(f"Unsupported compression type: {compression_type}")
                return compressed_data
                
        except Exception as e:
            logger.error(f"Error decompressing data with {compression_type}: {e}")
            return compressed_data
    
    @staticmethod
    def get_compression_ratio(original_size: int, compressed_size: int) -> float:
        """
        计算压缩比
        
        Args:
            original_size: 原始数据大小
            compressed_size: 压缩后数据大小
        
        Returns:
            压缩比 (0-1之间，越小压缩效果越好)
        """
        if original_size == 0:
            return 1.0
        return compressed_size / original_size
    
    @staticmethod
    def get_compression_info(compression_type: CompressionType) -> Dict[str, Any]:
        """
        获取压缩算法信息
        
        Args:
            compression_type: 压缩算法
        
        Returns:
            压缩算法信息字典
        """
        compression_info = {
            CompressionType.NONE: {
                "name": "No Compression",
                "speed": "fastest",
                "ratio": "worst",
                "cpu_usage": "minimal"
            },
            CompressionType.GZIP: {
                "name": "GZIP",
                "speed": "medium",
                "ratio": "good", 
                "cpu_usage": "medium"
            },
            CompressionType.BZIP2: {
                "name": "BZIP2",
                "speed": "slow",
                "ratio": "better",
                "cpu_usage": "high"
            },
            CompressionType.LZMA: {
                "name": "LZMA/XZ",
                "speed": "slowest",
                "ratio": "best",
                "cpu_usage": "highest"
            },
            CompressionType.SNAPPY: {
                "name": "Snappy",
                "speed": "fast",
                "ratio": "fair",
                "cpu_usage": "low"
            }
        }
        
        return compression_info.get(compression_type, {})
    
    @staticmethod
    def benchmark_compression(data: bytes, 
                            algorithms: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
        """
        压缩算法基准测试
        
        Args:
            data: 测试数据
            algorithms: 要测试的算法列表，默认测试所有
        
        Returns:
            基准测试结果
        """
        import time
        
        if algorithms is None:
            algorithms = list(CompressionType)
        
        results = {}
        original_size = len(data)
        
        for comp_type in algorithms:
            try:
                # 压缩测试
                start_time = time.time()
                compressed_data = CompressionUtils.compress_data(data, comp_type)
                compress_time = time.time() - start_time
                
                # 解压测试
                start_time = time.time()
                decompressed_data = CompressionUtils.decompress_data(compressed_data, comp_type)
                decompress_time = time.time() - start_time
                
                # 验证数据完整性
                data_valid = (data == decompressed_data)
                
                results[comp_type.value] = {
                    "original_size": original_size,
                    "compressed_size": len(compressed_data),
                    "compression_ratio": CompressionUtils.get_compression_ratio(
                        original_size, len(compressed_data)
                    ),
                    "compress_time": compress_time,
                    "decompress_time": decompress_time,
                    "total_time": compress_time + decompress_time,
                    "data_valid": data_valid,
                    "space_saved_percent": (1 - len(compressed_data) / original_size) * 100
                }
                
            except Exception as e:
                logger.error(f"Error benchmarking {comp_type}: {e}")
                results[comp_type.value] = {
                    "error": str(e)
                }
        
        return results


def choose_optimal_compression(data_size: int,
                              priority: str = "balanced") -> CompressionType:
    """
    根据数据大小和优先级选择最优压缩算法
    
    Args:
        data_size: 数据大小（字节）
        priority: 优化优先级
                 - "speed": 优先考虑速度
                 - "ratio": 优先考虑压缩比
                 - "balanced": 平衡速度和压缩比
    
    Returns:
        推荐的压缩算法
    """
    # 小数据不压缩
    if data_size < 1024:  # 1KB
        return CompressionType.NONE
    
    # 根据优先级选择
    if priority == "speed":
        if data_size < 1024 * 1024:  # 1MB
            return CompressionType.SNAPPY
        else:
            return CompressionType.GZIP
    elif priority == "ratio":
        if data_size > 10 * 1024 * 1024:  # 10MB
            return CompressionType.LZMA
        else:
            return CompressionType.BZIP2
    else:  # balanced
        if data_size < 10 * 1024 * 1024:  # 10MB
            return CompressionType.SNAPPY
        else:
            return CompressionType.GZIP
