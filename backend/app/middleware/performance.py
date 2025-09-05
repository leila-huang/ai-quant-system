"""
性能监控中间件

监控API请求性能，收集响应时间、内存使用等指标。
"""

import time
import psutil
import os
from typing import Callable, Dict, Any
from collections import defaultdict
from threading import Lock
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """性能监控中间件"""
    
    def __init__(self, app, warning_threshold: float = 1.0, enable_memory_monitoring: bool = True):
        super().__init__(app)
        self.warning_threshold = warning_threshold  # 响应时间警告阈值（秒）
        self.enable_memory_monitoring = enable_memory_monitoring
        
        # 性能统计数据
        self._stats = defaultdict(list)
        self._stats_lock = Lock()
        
        # 当前进程
        self._process = psutil.Process(os.getpid())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求并监控性能"""
        request_id = getattr(request.state, "request_id", "unknown")
        path = request.url.path
        method = request.method
        
        # 记录开始时间和内存使用
        start_time = time.time()
        start_memory = None
        
        if self.enable_memory_monitoring:
            try:
                memory_info = self._process.memory_info()
                start_memory = memory_info.rss  # 常驻内存大小
            except Exception:
                start_memory = None
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算性能指标
            end_time = time.time()
            response_time = end_time - start_time
            
            memory_diff = None
            current_memory = None
            
            if self.enable_memory_monitoring and start_memory:
                try:
                    memory_info = self._process.memory_info()
                    current_memory = memory_info.rss
                    memory_diff = current_memory - start_memory
                except Exception:
                    pass
            
            # 记录性能数据
            perf_data = {
                "path": path,
                "method": method,
                "status_code": response.status_code,
                "response_time": response_time,
                "timestamp": end_time,
                "request_id": request_id
            }
            
            if memory_diff is not None:
                perf_data["memory_diff"] = memory_diff
                perf_data["current_memory"] = current_memory
            
            # 存储统计数据
            with self._stats_lock:
                self._stats[f"{method} {path}"].append({
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "timestamp": end_time
                })
                
                # 保留最近1000条记录
                if len(self._stats[f"{method} {path}"]) > 1000:
                    self._stats[f"{method} {path}"] = self._stats[f"{method} {path}"][-1000:]
            
            # 添加性能头部信息
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            
            if memory_diff is not None:
                response.headers["X-Memory-Usage"] = f"{memory_diff / 1024 / 1024:.2f}MB"
            
            # 性能警告
            if response_time > self.warning_threshold:
                logger.warning(
                    f"Slow request detected: {method} {path} "
                    f"- Time: {response_time:.3f}s "
                    f"- Threshold: {self.warning_threshold}s "
                    f"- Request ID: {request_id}"
                )
            
            # 记录性能日志（只在调试模式或慢请求时记录）
            if response_time > self.warning_threshold:
                log_message = (
                    f"Performance: {method} {path} "
                    f"- Time: {response_time:.3f}s "
                    f"- Status: {response.status_code}"
                )
                
                if memory_diff is not None:
                    log_message += f" - Memory: {memory_diff / 1024 / 1024:.2f}MB"
                
                logger.info(log_message)
            
            return response
            
        except Exception as exc:
            # 即使在异常情况下也要记录性能数据
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.error(
                f"Request failed with performance data: {method} {path} "
                f"- Time: {response_time:.3f}s "
                f"- Exception: {type(exc).__name__} "
                f"- Request ID: {request_id}"
            )
            
            raise
    
    def get_stats(self, path_pattern: str = None) -> Dict[str, Any]:
        """获取性能统计数据"""
        with self._stats_lock:
            if path_pattern:
                # 返回特定路径的统计
                stats_data = {}
                for key, values in self._stats.items():
                    if path_pattern in key:
                        stats_data[key] = self._calculate_stats(values)
                return stats_data
            else:
                # 返回所有路径的统计
                return {key: self._calculate_stats(values) for key, values in self._stats.items()}
    
    def _calculate_stats(self, values: list) -> Dict[str, Any]:
        """计算统计指标"""
        if not values:
            return {}
        
        response_times = [v["response_time"] for v in values]
        status_codes = [v["status_code"] for v in values]
        
        # 基本统计
        stats = {
            "count": len(values),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "success_rate": len([s for s in status_codes if 200 <= s < 300]) / len(status_codes) * 100
        }
        
        # 百分位数
        sorted_times = sorted(response_times)
        stats["p50_response_time"] = self._percentile(sorted_times, 50)
        stats["p95_response_time"] = self._percentile(sorted_times, 95)
        stats["p99_response_time"] = self._percentile(sorted_times, 99)
        
        # 状态码分布
        status_distribution = {}
        for status_code in status_codes:
            status_range = f"{status_code // 100}xx"
            status_distribution[status_range] = status_distribution.get(status_range, 0) + 1
        
        stats["status_distribution"] = status_distribution
        
        return stats
    
    def _percentile(self, sorted_list: list, percentile: float) -> float:
        """计算百分位数"""
        if not sorted_list:
            return 0.0
        
        k = (len(sorted_list) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        
        if f == len(sorted_list) - 1:
            return sorted_list[f]
        
        return sorted_list[f] * (1 - c) + sorted_list[f + 1] * c
    
    def clear_stats(self):
        """清空统计数据"""
        with self._stats_lock:
            self._stats.clear()
            
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统资源使用情况"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # 进程内存使用
            process_memory = self._process.memory_info()
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "process_memory": {
                    "rss": process_memory.rss,  # 常驻内存
                    "vms": process_memory.vms,  # 虚拟内存
                }
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
