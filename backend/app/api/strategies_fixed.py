"""
策略管理API路由

提供策略的创建、查询、更新、删除、启停等完整功能。
与前端Strategy页面数据格式匹配，重用现有数据库模型。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from backend.src.database.models import Strategy, StrategyStatus
from backend.src.database.connection import DatabaseManager
from backend.app.core.exceptions import APIException, BusinessException, DatabaseException


router = APIRouter()

# 数据库管理器实例
db_manager = DatabaseManager()


# === Pydantic 请求/响应模型定义 ===

class StrategyCreateRequest(BaseModel):
    """策略创建请求模型"""
    name: str = Field(..., min_length=1, max_length=100, description="策略名称")
    description: Optional[str] = Field(None, max_length=1000, description="策略描述")
    type: str = Field(..., description="策略类型")
    config: Dict[str, Any] = Field(default_factory=dict, description="策略参数配置")
    code: Optional[str] = Field(None, description="策略代码")
    version: str = Field(default="1.0.0", description="策略版本")
    max_position_size: float = Field(default=10000.0, gt=0, description="最大仓位金额")
    max_daily_loss: float = Field(default=1000.0, gt=0, description="最大日损失")
    max_drawdown: float = Field(default=0.20, gt=0, le=1.0, description="最大回撤比例")


class StrategyUpdateRequest(BaseModel):
    """策略更新请求模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="策略名称")
    description: Optional[str] = Field(None, max_length=1000, description="策略描述")
    type: Optional[str] = Field(None, description="策略类型")
    config: Optional[Dict[str, Any]] = Field(None, description="策略参数配置")
    code: Optional[str] = Field(None, description="策略代码")
    version: Optional[str] = Field(None, description="策略版本")
    max_position_size: Optional[float] = Field(None, gt=0, description="最大仓位金额")
    max_daily_loss: Optional[float] = Field(None, gt=0, description="最大日损失")
    max_drawdown: Optional[float] = Field(None, gt=0, le=1.0, description="最大回撤比例")


class StrategyResponse(BaseModel):
    """策略响应模型 - 匹配前端Strategy接口"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str  # 将数据库id转为字符串以匹配前端期望
    name: str
    type: str = Field(default="custom", description="策略类型")
    status: str  # 将StrategyStatus枚举转换为字符串
    description: Optional[str] = None
    
    # 绩效指标（模拟数据，后续可从回测结果计算）
    totalReturn: float = Field(default=0.0, description="总收益率")
    annualReturn: float = Field(default=0.0, description="年化收益率")
    maxDrawdown: float = Field(default=0.0, description="最大回撤")
    sharpeRatio: float = Field(default=0.0, description="夏普比率")
    
    # 时间字段
    createdAt: str  # ISO格式字符串
    lastRunAt: Optional[str] = None  # ISO格式字符串
    
    # 配置参数
    parameters: Dict[str, Any] = Field(default_factory=dict, description="策略参数")
    
    # 风控参数
    max_position_size: float = Field(default=10000.0, description="最大仓位金额")
    max_daily_loss: float = Field(default=1000.0, description="最大日损失")
    max_drawdown_limit: float = Field(default=0.20, description="最大回撤限制")
    
    # 版本和代码信息
    version: str = Field(default="1.0.0", description="版本号")
    code: Optional[str] = None


class StrategyListResponse(BaseModel):
    """策略列表响应模型"""
    strategies: List[StrategyResponse]
    total_count: int
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)


class StrategyActionResponse(BaseModel):
    """策略操作响应模型"""
    success: bool
    message: str
    strategy_id: str
    new_status: Optional[str] = None


# === 数据库操作函数 ===

def db_strategy_to_response(db_strategy: Strategy) -> StrategyResponse:
    """将数据库Strategy对象转换为响应模型"""
    return StrategyResponse(
        id=str(db_strategy.id),
        name=db_strategy.name,
        type=db_strategy.config.get("type", "custom") if db_strategy.config else "custom",
        status=db_strategy.status.value,
        description=db_strategy.description,
        # 绩效数据暂时使用模拟值，后续可从策略运行结果计算
        totalReturn=0.0,
        annualReturn=0.0,
        maxDrawdown=-abs(float(db_strategy.max_drawdown or 0.0)),
        sharpeRatio=0.0,
        createdAt=db_strategy.created_at.isoformat() if db_strategy.created_at else datetime.now().isoformat(),
        lastRunAt=db_strategy.last_run_at.isoformat() if db_strategy.last_run_at else None,
        parameters=db_strategy.config or {},
        max_position_size=float(db_strategy.max_position_size or 10000.0),
        max_daily_loss=float(db_strategy.max_daily_loss or 1000.0),
        max_drawdown_limit=float(db_strategy.max_drawdown or 0.20),
        version=db_strategy.version or "1.0.0",
        code=db_strategy.code
    )


# === API 路由端点 ===

@router.get("/", response_model=StrategyListResponse, summary="查询策略列表")
async def list_strategies(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="按状态过滤"),
    search: Optional[str] = Query(None, description="按名称搜索")
):
    """
    获取策略列表
    
    支持分页、按状态过滤、按名称搜索等功能。
    """
    try:
        with db_manager.get_session() as db:
            # 构建查询
            query = db.query(Strategy)
            
            # 临时使用固定user_id=1（P3阶段会实现用户认证）
            query = query.filter(Strategy.user_id == 1)
            
            # 状态过滤
            if status:
                try:
                    status_enum = StrategyStatus(status)
                    query = query.filter(Strategy.status == status_enum)
                except ValueError:
                    raise BusinessException(f"无效的状态值: {status}")
            
            # 名称搜索
            if search:
                query = query.filter(Strategy.name.contains(search))
            
            # 总数统计
            total_count = query.count()
            
            # 分页查询
            offset = (page - 1) * page_size
            strategies = query.order_by(Strategy.created_at.desc()).offset(offset).limit(page_size).all()
            
            # 转换为响应格式
            strategy_responses = [db_strategy_to_response(strategy) for strategy in strategies]
            
            return StrategyListResponse(
                strategies=strategy_responses,
                total_count=total_count,
                page=page,
                page_size=page_size
            )
        
    except Exception as e:
        if isinstance(e, (BusinessException, APIException)):
            raise
        raise DatabaseException(f"查询策略列表失败: {str(e)}")


@router.get("/{strategy_id}", response_model=StrategyResponse, summary="获取单个策略详情")
async def get_strategy(strategy_id: int):
    """
    根据ID获取策略详情
    """
    try:
        with db_manager.get_session() as db:
            strategy = db.query(Strategy).filter(
                Strategy.id == strategy_id,
                Strategy.user_id == 1  # 临时固定user_id
            ).first()
            
            if not strategy:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"策略ID {strategy_id} 不存在"
                )
            
            return db_strategy_to_response(strategy)
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseException(f"获取策略详情失败: {str(e)}")


@router.post("/", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED, summary="创建新策略")
async def create_strategy(strategy_data: StrategyCreateRequest):
    """
    创建新的交易策略
    """
    try:
        with db_manager.get_session() as db:
            # 检查策略名称是否重复
            existing = db.query(Strategy).filter(
                Strategy.name == strategy_data.name,
                Strategy.user_id == 1  # 临时固定user_id
            ).first()
            
            if existing:
                raise BusinessException(f"策略名称 '{strategy_data.name}' 已存在")
            
            # 将type添加到config中
            config = strategy_data.config.copy()
            config["type"] = strategy_data.type
            
            # 创建策略对象
            new_strategy = Strategy(
                user_id=1,  # 临时固定user_id
                name=strategy_data.name,
                description=strategy_data.description,
                status=StrategyStatus.STOPPED,  # 新策略默认停止状态
                config=config,
                code=strategy_data.code,
                version=strategy_data.version,
                max_position_size=Decimal(str(strategy_data.max_position_size)),
                max_daily_loss=Decimal(str(strategy_data.max_daily_loss)),
                max_drawdown=Decimal(str(strategy_data.max_drawdown))
            )
            
            db.add(new_strategy)
            db.commit()
            db.refresh(new_strategy)
            
            return db_strategy_to_response(new_strategy)
        
    except Exception as e:
        if isinstance(e, (BusinessException, APIException)):
            raise
        raise DatabaseException(f"创建策略失败: {str(e)}")


@router.put("/{strategy_id}", response_model=StrategyResponse, summary="更新策略")
async def update_strategy(
    strategy_id: int,
    strategy_data: StrategyUpdateRequest
):
    """
    更新策略信息
    """
    try:
        with db_manager.get_session() as db:
            strategy = db.query(Strategy).filter(
                Strategy.id == strategy_id,
                Strategy.user_id == 1  # 临时固定user_id
            ).first()
            
            if not strategy:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"策略ID {strategy_id} 不存在"
                )
            
            # 检查策略是否正在运行（运行中的策略不能修改关键参数）
            if strategy.status == StrategyStatus.ACTIVE:
                restricted_fields = {'config', 'code', 'max_position_size', 'max_daily_loss', 'max_drawdown'}
                update_dict = strategy_data.model_dump(exclude_unset=True)
                if any(field in update_dict for field in restricted_fields):
                    raise BusinessException("运行中的策略不能修改关键参数，请先停止策略")
            
            # 检查名称重复
            if strategy_data.name and strategy_data.name != strategy.name:
                existing = db.query(Strategy).filter(
                    Strategy.name == strategy_data.name,
                    Strategy.user_id == 1,
                    Strategy.id != strategy_id
                ).first()
                if existing:
                    raise BusinessException(f"策略名称 '{strategy_data.name}' 已存在")
            
            # 更新字段
            update_dict = strategy_data.model_dump(exclude_unset=True)
            
            for field, value in update_dict.items():
                if field == 'config' and value is not None:
                    # 保持现有type，如果有的话
                    if strategy_data.type:
                        value = value.copy()
                        value["type"] = strategy_data.type
                    elif strategy.config and "type" in strategy.config:
                        value = value.copy()
                        value["type"] = strategy.config["type"]
                elif field in {'max_position_size', 'max_daily_loss', 'max_drawdown'} and value is not None:
                    value = Decimal(str(value))
                
                setattr(strategy, field, value)
            
            # 更新type到config
            if strategy_data.type:
                if not strategy.config:
                    strategy.config = {}
                strategy.config["type"] = strategy_data.type
            
            # 更新时间戳
            strategy.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(strategy)
            
            return db_strategy_to_response(strategy)
        
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, (BusinessException, APIException)):
            raise
        raise DatabaseException(f"更新策略失败: {str(e)}")


@router.delete("/{strategy_id}", response_model=StrategyActionResponse, summary="删除策略")
async def delete_strategy(strategy_id: int):
    """
    删除策略
    """
    try:
        with db_manager.get_session() as db:
            strategy = db.query(Strategy).filter(
                Strategy.id == strategy_id,
                Strategy.user_id == 1  # 临时固定user_id
            ).first()
            
            if not strategy:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"策略ID {strategy_id} 不存在"
                )
            
            # 检查策略状态（运行中的策略不能删除）
            if strategy.status == StrategyStatus.ACTIVE:
                raise BusinessException("无法删除运行中的策略，请先停止策略")
            
            db.delete(strategy)
            db.commit()
            
            return StrategyActionResponse(
                success=True,
                message=f"策略 '{strategy.name}' 已成功删除",
                strategy_id=str(strategy_id)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, (BusinessException, APIException)):
            raise
        raise DatabaseException(f"删除策略失败: {str(e)}")


@router.post("/{strategy_id}/start", response_model=StrategyActionResponse, summary="启动策略")
async def start_strategy(strategy_id: int):
    """
    启动策略
    """
    try:
        with db_manager.get_session() as db:
            strategy = db.query(Strategy).filter(
                Strategy.id == strategy_id,
                Strategy.user_id == 1  # 临时固定user_id
            ).first()
            
            if not strategy:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"策略ID {strategy_id} 不存在"
                )
            
            # 检查当前状态
            if strategy.status == StrategyStatus.ACTIVE:
                raise BusinessException("策略已经在运行中")
            
            # 基本验证
            if not strategy.code:
                raise BusinessException("策略代码为空，无法启动")
            
            if not strategy.config:
                raise BusinessException("策略配置为空，无法启动")
            
            # 更新状态
            strategy.status = StrategyStatus.ACTIVE
            strategy.last_run_at = datetime.utcnow()
            strategy.updated_at = datetime.utcnow()
            
            db.commit()
            
            return StrategyActionResponse(
                success=True,
                message=f"策略 '{strategy.name}' 已成功启动",
                strategy_id=str(strategy_id),
                new_status="active"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, (BusinessException, APIException)):
            raise
        raise DatabaseException(f"启动策略失败: {str(e)}")


@router.post("/{strategy_id}/stop", response_model=StrategyActionResponse, summary="停止策略")
async def stop_strategy(strategy_id: int):
    """
    停止策略
    """
    try:
        with db_manager.get_session() as db:
            strategy = db.query(Strategy).filter(
                Strategy.id == strategy_id,
                Strategy.user_id == 1  # 临时固定user_id
            ).first()
            
            if not strategy:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"策略ID {strategy_id} 不存在"
                )
            
            # 检查当前状态
            if strategy.status == StrategyStatus.STOPPED:
                raise BusinessException("策略已经是停止状态")
            
            # 更新状态
            old_status = strategy.status.value
            strategy.status = StrategyStatus.STOPPED
            strategy.updated_at = datetime.utcnow()
            
            db.commit()
            
            return StrategyActionResponse(
                success=True,
                message=f"策略 '{strategy.name}' 已从 '{old_status}' 状态停止",
                strategy_id=str(strategy_id),
                new_status="stopped"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, (BusinessException, APIException)):
            raise
        raise DatabaseException(f"停止策略失败: {str(e)}")


@router.post("/{strategy_id}/pause", response_model=StrategyActionResponse, summary="暂停策略")
async def pause_strategy(strategy_id: int):
    """
    暂停策略
    """
    try:
        with db_manager.get_session() as db:
            strategy = db.query(Strategy).filter(
                Strategy.id == strategy_id,
                Strategy.user_id == 1  # 临时固定user_id
            ).first()
            
            if not strategy:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"策略ID {strategy_id} 不存在"
                )
            
            # 只有运行中的策略才能暂停
            if strategy.status != StrategyStatus.ACTIVE:
                raise BusinessException("只有运行中的策略才能暂停")
            
            # 更新状态
            strategy.status = StrategyStatus.PAUSED
            strategy.updated_at = datetime.utcnow()
            
            db.commit()
            
            return StrategyActionResponse(
                success=True,
                message=f"策略 '{strategy.name}' 已暂停",
                strategy_id=str(strategy_id),
                new_status="paused"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, (BusinessException, APIException)):
            raise
        raise DatabaseException(f"暂停策略失败: {str(e)}")


# === 统计信息端点 ===

@router.get("/stats/summary", summary="策略统计概览")
async def get_strategy_stats():
    """
    获取策略统计概览信息
    """
    try:
        with db_manager.get_session() as db:
            # 统计各状态的策略数量
            total = db.query(Strategy).filter(Strategy.user_id == 1).count()
            active = db.query(Strategy).filter(
                Strategy.user_id == 1,
                Strategy.status == StrategyStatus.ACTIVE
            ).count()
            paused = db.query(Strategy).filter(
                Strategy.user_id == 1, 
                Strategy.status == StrategyStatus.PAUSED
            ).count()
            stopped = db.query(Strategy).filter(
                Strategy.user_id == 1,
                Strategy.status == StrategyStatus.STOPPED
            ).count()
            
            return {
                "total_strategies": total,
                "active_strategies": active,
                "paused_strategies": paused,
                "stopped_strategies": stopped,
                "last_updated": datetime.now().isoformat()
            }
        
    except Exception as e:
        raise DatabaseException(f"获取策略统计失败: {str(e)}")
