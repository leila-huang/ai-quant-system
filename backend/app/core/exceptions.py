"""
FastAPI全局异常处理器

统一处理各类异常，提供标准化的错误响应格式。
"""

import traceback
from typing import Any, Dict, Optional, Union
from datetime import datetime

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from pydantic import ValidationError as PydanticValidationError
from loguru import logger

from backend.src.database.crud import CRUDError


class APIException(HTTPException):
    """自定义API异常基类"""
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Any = None,
        headers: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        super().__init__(status_code, detail, headers)
        self.error_code = error_code


class BusinessException(APIException):
    """业务逻辑异常"""
    
    def __init__(self, detail: str, error_code: str = "BUSINESS_ERROR"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code
        )


class DatabaseException(APIException):
    """数据库异常"""
    
    def __init__(self, detail: str = "数据库操作失败", error_code: str = "DATABASE_ERROR"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code
        )


class DataValidationException(APIException):
    """数据验证异常"""
    
    def __init__(self, detail: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code
        )


class AuthenticationException(APIException):
    """认证异常"""
    
    def __init__(self, detail: str = "认证失败", error_code: str = "AUTH_ERROR"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code
        )


class AuthorizationException(APIException):
    """授权异常"""
    
    def __init__(self, detail: str = "权限不足", error_code: str = "PERMISSION_ERROR"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code=error_code
        )


class RateLimitException(APIException):
    """访问频率限制异常"""
    
    def __init__(self, detail: str = "请求频率过高", error_code: str = "RATE_LIMIT_ERROR"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code=error_code
        )


class ExternalServiceException(APIException):
    """外部服务异常"""
    
    def __init__(self, detail: str = "外部服务不可用", error_code: str = "EXTERNAL_SERVICE_ERROR"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code=error_code
        )


def create_error_response(
    status_code: int,
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建标准化错误响应"""
    error_response = {
        "success": False,
        "error": {
            "code": error_code or f"HTTP_{status_code}",
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
    }
    
    if details:
        error_response["error"]["details"] = details
    
    if request_id:
        error_response["error"]["request_id"] = request_id
    
    return error_response


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    error_code = None
    if isinstance(exc, APIException):
        error_code = exc.error_code
    
    logger.error(
        f"HTTP Exception: {exc.status_code} - {exc.detail} "
        f"- Path: {request.url.path} - Request ID: {request_id}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_code=error_code,
            request_id=request_id
        )
    )


async def starlette_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Starlette HTTP异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        f"Starlette HTTP Exception: {exc.status_code} - {exc.detail} "
        f"- Path: {request.url.path} - Request ID: {request_id}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            request_id=request_id
        )
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """请求验证异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    logger.error(
        f"Validation Error: {error_details} "
        f"- Path: {request.url.path} - Request ID: {request_id}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="请求数据验证失败",
            error_code="VALIDATION_ERROR",
            details={"validation_errors": error_details},
            request_id=request_id
        )
    )


async def pydantic_validation_exception_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """Pydantic验证异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    logger.error(
        f"Pydantic Validation Error: {error_details} "
        f"- Path: {request.url.path} - Request ID: {request_id}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="数据模型验证失败",
            error_code="MODEL_VALIDATION_ERROR",
            details={"validation_errors": error_details},
            request_id=request_id
        )
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """SQLAlchemy数据库异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    error_code = "DATABASE_ERROR"
    message = "数据库操作失败"
    
    if isinstance(exc, IntegrityError):
        error_code = "DATABASE_INTEGRITY_ERROR"
        message = "数据完整性约束违反"
    elif isinstance(exc, OperationalError):
        error_code = "DATABASE_OPERATIONAL_ERROR"
        message = "数据库操作错误"
    
    logger.error(
        f"SQLAlchemy Error: {type(exc).__name__} - {str(exc)} "
        f"- Path: {request.url.path} - Request ID: {request_id}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_code=error_code,
            request_id=request_id
        )
    )


async def crud_exception_handler(request: Request, exc: CRUDError) -> JSONResponse:
    """CRUD操作异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        f"CRUD Error: {str(exc)} "
        f"- Path: {request.url.path} - Request ID: {request_id}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="数据操作失败",
            error_code="CRUD_ERROR",
            request_id=request_id
        )
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        f"Unhandled Exception: {type(exc).__name__} - {str(exc)} "
        f"- Path: {request.url.path} - Request ID: {request_id}",
        exc_info=True
    )
    
    # 在开发环境返回详细错误信息
    details = None
    try:
        from backend.app.core.config import settings
        if settings.DEBUG:
            details = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc().split("\n")
            }
    except Exception:
        pass
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="服务器内部错误",
            error_code="INTERNAL_SERVER_ERROR",
            details=details,
            request_id=request_id
        )
    )


def setup_exception_handlers(app):
    """设置异常处理器"""
    # HTTP异常
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_exception_handler)
    
    # 验证异常
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(PydanticValidationError, pydantic_validation_exception_handler)
    
    # 数据库异常
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    app.add_exception_handler(CRUDError, crud_exception_handler)
    
    # 通用异常
    app.add_exception_handler(Exception, general_exception_handler)
