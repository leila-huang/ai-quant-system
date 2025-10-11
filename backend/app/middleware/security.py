"""
安全头中间件

添加安全相关的HTTP响应头，提高应用安全性。
"""

from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全头中间件"""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """添加安全响应头"""
        response = await call_next(request)
        
        # 安全头配置
        security_headers = {
            # 防止XSS攻击
            "X-Content-Type-Options": "nosniff",
            
            # 防止点击劫持
            "X-Frame-Options": "DENY",
            
            # XSS保护
            "X-XSS-Protection": "1; mode=block",
            
            # 引用者政策
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # 权限政策
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            
            # 内容安全策略 (开发环境相对宽松)
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; "
                "style-src 'self' 'unsafe-inline' https:; "
                "img-src 'self' data: https:; "
                "font-src 'self' data: https:; "
                "connect-src 'self' ws: wss: https:; "
                "frame-ancestors 'none';"
            ),
            
            # HSTS (仅在HTTPS时启用)
            # "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
        
        # 添加安全头
        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value
        
        # 移除可能泄露服务器信息的头
        if "Server" in response.headers:
            del response.headers["Server"]
        
        return response


def get_security_headers_config() -> dict:
    """获取安全头配置信息"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY", 
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none';"
        )
    }
