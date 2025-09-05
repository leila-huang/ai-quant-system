"""
请求追踪中间件

为每个HTTP请求生成唯一ID，用于日志跟踪和问题排查。
"""

import uuid
import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """请求追踪中间件"""
    
    def __init__(self, app, request_id_header: str = "X-Request-ID"):
        super().__init__(app)
        self.request_id_header = request_id_header
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求"""
        # 生成请求ID
        request_id = request.headers.get(self.request_id_header) or str(uuid.uuid4())
        
        # 将请求ID存储到请求状态中
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取客户端IP
        client_ip = self._get_client_ip(request)
        
        # 记录请求开始日志
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"- Client: {client_ip} - Request ID: {request_id}"
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 添加响应头
            response.headers[self.request_id_header] = request_id
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
            
            # 记录请求完成日志
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} "
                f"- Time: {process_time:.3f}s "
                f"- Request ID: {request_id}"
            )
            
            return response
            
        except Exception as exc:
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录请求异常日志
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Exception: {type(exc).__name__} - {str(exc)} "
                f"- Time: {process_time:.3f}s "
                f"- Request ID: {request_id}",
                exc_info=True
            )
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        # 检查代理头
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # 回退到直接连接IP
        return request.client.host if request.client else "unknown"
