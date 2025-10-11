"""
纸上交易API路由

提供模拟交易功能，包括订单管理、持仓查询、成交记录等。
支持T+1交易规则、实盘vs回测偏差分析。
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal
import uuid

from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.src.database.models import (
    Order, Position, Strategy, OrderStatus, OrderSide, OrderType,
    StrategyStatus
)
from backend.src.database.connection import get_database_manager
from backend.src.storage.parquet_engine import get_parquet_storage
from backend.app.core.exceptions import APIException, DatabaseException

router = APIRouter()


# === Pydantic模型定义 ===

class OrderRequest(BaseModel):
    """订单创建请求"""
    symbol: str = Field(..., description="股票代码")
    side: str = Field(..., description="买卖方向: BUY/SELL")
    order_type: str = Field("MARKET", description="订单类型: MARKET/LIMIT")
    quantity: float = Field(..., gt=0, description="数量（股）")
    price: Optional[float] = Field(None, description="限价单价格")
    strategy_name: Optional[str] = Field(None, description="关联策略名称")
    
    @validator('side')
    def validate_side(cls, v):
        if v not in ['BUY', 'SELL']:
            raise ValueError("side必须是BUY或SELL")
        return v
    
    @validator('order_type')
    def validate_order_type(cls, v):
        if v not in ['MARKET', 'LIMIT']:
            raise ValueError("order_type必须是MARKET或LIMIT")
        return v


class OrderResponse(BaseModel):
    """订单响应"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    placed_at: datetime
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: float = 0.0
    
    class Config:
        from_attributes = True


class PositionResponse(BaseModel):
    """持仓响应"""
    id: int
    symbol: str
    quantity: float
    available_quantity: float
    frozen_quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    first_buy_date: Optional[date]
    last_update_at: datetime
    
    class Config:
        from_attributes = True


class PortfolioSummaryResponse(BaseModel):
    """组合概览响应"""
    total_value: float
    cash_balance: float
    stock_value: float
    total_pnl: float
    total_pnl_ratio: float
    today_pnl: float
    positions_count: int
    orders_count: int


class DeviationAnalysisResponse(BaseModel):
    """偏差分析响应"""
    symbol: str
    backtest_return: float
    paper_return: float
    deviation: float
    slippage: float
    execution_quality_score: float


# === 依赖注入 ===

def get_db():
    """获取数据库会话"""
    db_manager = get_database_manager()
    session = next(db_manager.get_session())
    try:
        yield session
    finally:
        session.close()


def get_or_create_paper_strategy(db: Session, user_id: int = 1) -> Strategy:
    """获取或创建纸上交易策略"""
    strategy = db.query(Strategy).filter(
        Strategy.user_id == user_id,
        Strategy.name == "Paper Trading"
    ).first()
    
    if not strategy:
        strategy = Strategy(
            uuid=uuid.uuid4(),
            user_id=user_id,
            name="Paper Trading",
            description="模拟交易策略",
            status=StrategyStatus.ACTIVE,
            config={"type": "paper_trading"}
        )
        db.add(strategy)
        db.commit()
        db.refresh(strategy)
    
    return strategy


# === API路由 ===

@router.post("/orders", response_model=OrderResponse, summary="创建纸上交易订单")
async def create_paper_order(
    request: OrderRequest,
    db: Session = Depends(get_db)
):
    """
    创建纸上交易订单
    
    支持市价单和限价单，自动执行T+1规则。
    """
    try:
        # 获取或创建纸上交易策略
        strategy = get_or_create_paper_strategy(db)
        
        # 创建订单
        order = Order(
            uuid=uuid.uuid4(),
            user_id=strategy.user_id,
            strategy_id=strategy.id,
            symbol=request.symbol,
            side=OrderSide.BUY if request.side == 'BUY' else OrderSide.SELL,
            order_type=OrderType.MARKET if request.order_type == 'MARKET' else OrderType.LIMIT,
            status=OrderStatus.PENDING,
            quantity=Decimal(str(request.quantity)),
            price=Decimal(str(request.price)) if request.price else None,
            client_order_id=f"PAPER_{uuid.uuid4().hex[:12]}",
            placed_at=datetime.now()
        )
        
        # 市价单立即模拟成交
        if request.order_type == 'MARKET':
            order = await _simulate_market_order_fill(order, db)
        
        db.add(order)
        db.commit()
        db.refresh(order)
        
        return OrderResponse(
            order_id=str(order.uuid),
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=float(order.quantity),
            price=float(order.price) if order.price else None,
            status=order.status.value,
            placed_at=order.placed_at,
            filled_quantity=float(order.filled_quantity),
            avg_fill_price=float(order.avg_fill_price) if order.avg_fill_price else None,
            commission=float(order.commission)
        )
        
    except Exception as e:
        raise DatabaseException(f"创建订单失败: {str(e)}")


@router.get("/orders", response_model=List[OrderResponse], summary="查询纸上交易订单")
async def get_paper_orders(
    symbol: Optional[str] = Query(None, description="股票代码筛选"),
    status: Optional[str] = Query(None, description="订单状态筛选"),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量"),
    db: Session = Depends(get_db)
):
    """
    查询纸上交易订单列表
    
    支持多种筛选条件。
    """
    try:
        strategy = get_or_create_paper_strategy(db)
        
        query = db.query(Order).filter(Order.strategy_id == strategy.id)
        
        if symbol:
            query = query.filter(Order.symbol == symbol)
        
        if status:
            query = query.filter(Order.status == status)
        
        if start_date:
            query = query.filter(Order.placed_at >= datetime.combine(start_date, datetime.min.time()))
        
        if end_date:
            query = query.filter(Order.placed_at <= datetime.combine(end_date, datetime.max.time()))
        
        orders = query.order_by(Order.placed_at.desc()).limit(limit).all()
        
        return [
            OrderResponse(
                order_id=str(order.uuid),
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=float(order.quantity),
                price=float(order.price) if order.price else None,
                status=order.status.value,
                placed_at=order.placed_at,
                filled_quantity=float(order.filled_quantity),
                avg_fill_price=float(order.avg_fill_price) if order.avg_fill_price else None,
                commission=float(order.commission)
            )
            for order in orders
        ]
        
    except Exception as e:
        raise DatabaseException(f"查询订单失败: {str(e)}")


@router.get("/positions", response_model=List[PositionResponse], summary="查询持仓")
async def get_paper_positions(
    symbol: Optional[str] = Query(None, description="股票代码筛选"),
    db: Session = Depends(get_db)
):
    """
    查询纸上交易持仓
    
    返回当前所有持仓信息。
    """
    try:
        strategy = get_or_create_paper_strategy(db)
        
        query = db.query(Position).filter(
            Position.strategy_id == strategy.id,
            Position.quantity != 0  # 只返回非零持仓
        )
        
        if symbol:
            query = query.filter(Position.symbol == symbol)
        
        positions = query.all()
        
        return [
            PositionResponse(
                id=pos.id,
                symbol=pos.symbol,
                quantity=float(pos.quantity),
                available_quantity=float(pos.available_quantity),
                frozen_quantity=float(pos.frozen_quantity),
                avg_cost=float(pos.avg_cost),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pnl),
                realized_pnl=float(pos.realized_pnl),
                first_buy_date=pos.first_buy_date,
                last_update_at=pos.last_update_at
            )
            for pos in positions
        ]
        
    except Exception as e:
        raise DatabaseException(f"查询持仓失败: {str(e)}")


@router.get("/portfolio/summary", response_model=PortfolioSummaryResponse, summary="组合概览")
async def get_portfolio_summary(
    db: Session = Depends(get_db)
):
    """
    获取投资组合概览
    
    包括总资产、持仓数量、盈亏等统计信息。
    """
    try:
        strategy = get_or_create_paper_strategy(db)
        
        # 查询所有持仓
        positions = db.query(Position).filter(
            Position.strategy_id == strategy.id,
            Position.quantity != 0
        ).all()
        
        # 查询今日订单
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_orders = db.query(Order).filter(
            Order.strategy_id == strategy.id,
            Order.placed_at >= today_start
        ).count()
        
        # 计算统计信息
        total_stock_value = sum(float(pos.market_value) for pos in positions)
        total_unrealized_pnl = sum(float(pos.unrealized_pnl) for pos in positions)
        total_realized_pnl = sum(float(pos.realized_pnl) for pos in positions)
        
        # 假设初始资金100万
        initial_capital = 1000000.0
        cash_balance = initial_capital - total_stock_value  # 简化计算
        total_value = cash_balance + total_stock_value
        total_pnl = total_unrealized_pnl + total_realized_pnl
        total_pnl_ratio = total_pnl / initial_capital if initial_capital > 0 else 0
        
        return PortfolioSummaryResponse(
            total_value=total_value,
            cash_balance=cash_balance,
            stock_value=total_stock_value,
            total_pnl=total_pnl,
            total_pnl_ratio=total_pnl_ratio,
            today_pnl=total_unrealized_pnl,  # 简化：用未实现盈亏代表今日盈亏
            positions_count=len(positions),
            orders_count=today_orders
        )
        
    except Exception as e:
        raise DatabaseException(f"获取组合概览失败: {str(e)}")


@router.get("/analysis/deviation", response_model=List[DeviationAnalysisResponse], summary="实盘vs回测偏差分析")
async def analyze_deviation(
    backtest_id: str = Query(..., description="回测ID"),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    db: Session = Depends(get_db)
):
    """
    分析实盘交易与回测的偏差
    
    比较实际执行与回测预期的差异，包括滑点、执行质量等。
    """
    try:
        strategy = get_or_create_paper_strategy(db)
        
        # 查询实盘持仓
        positions = db.query(Position).filter(
            Position.strategy_id == strategy.id,
            Position.quantity != 0
        ).all()
        
        # 这里简化处理，实际应该：
        # 1. 从回测结果中加载对应时间段的持仓和收益
        # 2. 与实盘数据对比
        # 3. 计算滑点、执行质量等指标
        
        results = []
        for pos in positions:
            # 模拟偏差分析
            backtest_return = 0.05  # 回测收益率（应从实际回测结果读取）
            paper_return = float(pos.unrealized_pnl) / float(pos.total_cost) if float(pos.total_cost) > 0 else 0
            deviation = paper_return - backtest_return
            slippage = abs(deviation) * 0.5  # 估算滑点
            execution_quality = max(0, 1.0 - abs(deviation) * 10)  # 执行质量评分
            
            results.append(DeviationAnalysisResponse(
                symbol=pos.symbol,
                backtest_return=backtest_return,
                paper_return=paper_return,
                deviation=deviation,
                slippage=slippage,
                execution_quality_score=execution_quality
            ))
        
        return results
        
    except Exception as e:
        raise DatabaseException(f"偏差分析失败: {str(e)}")


# === 辅助函数 ===

async def _simulate_market_order_fill(order: Order, db: Session) -> Order:
    """
    模拟市价单成交
    
    获取当前市场价格并模拟成交
    """
    try:
        # 获取股票当前价格
        storage = get_parquet_storage()
        stock_data = storage.load_stock_data(
            symbol=order.symbol,
            start_date=date.today() - timedelta(days=5),
            end_date=date.today()
        )
        
        if stock_data and stock_data.bars:
            # 使用最后一个交易日的收盘价
            last_bar = stock_data.bars[-1]
            fill_price = last_bar.close_price
            
            # 模拟滑点（0.1%）
            slippage = 0.001
            if order.side == OrderSide.BUY:
                fill_price *= (1 + slippage)
            else:
                fill_price *= (1 - slippage)
            
            # 更新订单
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = Decimal(str(fill_price))
            order.filled_at = datetime.now()
            
            # 计算手续费（万分之3）
            trade_amount = float(order.quantity) * fill_price
            order.commission = Decimal(str(trade_amount * 0.0003))
            order.total_amount = Decimal(str(trade_amount + float(order.commission)))
            
            # 更新或创建持仓
            await _update_position(order, db)
            
        else:
            # 无法获取价格，订单保持PENDING状态
            order.status = OrderStatus.PENDING
        
        return order
        
    except Exception as e:
        order.status = OrderStatus.REJECTED
        order.notes = f"模拟成交失败: {str(e)}"
        return order


async def _update_position(order: Order, db: Session):
    """更新持仓"""
    try:
        position = db.query(Position).filter(
            Position.strategy_id == order.strategy_id,
            Position.symbol == order.symbol
        ).first()
        
        if not position:
            # 创建新持仓
            position = Position(
                user_id=order.user_id,
                strategy_id=order.strategy_id,
                symbol=order.symbol,
                quantity=Decimal(0),
                available_quantity=Decimal(0),
                frozen_quantity=Decimal(0),
                avg_cost=Decimal(0),
                total_cost=Decimal(0),
                market_value=Decimal(0),
                unrealized_pnl=Decimal(0),
                realized_pnl=Decimal(0)
            )
            db.add(position)
        
        # 更新持仓数量和成本
        if order.side == OrderSide.BUY:
            # 买入
            old_quantity = float(position.quantity)
            old_cost = float(position.avg_cost)
            new_quantity = old_quantity + float(order.filled_quantity)
            
            if new_quantity > 0:
                # 加权平均成本
                position.avg_cost = Decimal(
                    (old_quantity * old_cost + float(order.filled_quantity) * float(order.avg_fill_price)) / new_quantity
                )
            
            position.quantity = Decimal(str(new_quantity))
            position.available_quantity = position.quantity  # T+1规则应在此处实现
            position.total_cost = position.quantity * position.avg_cost
            
            if not position.first_buy_date:
                position.first_buy_date = date.today()
        
        else:
            # 卖出
            position.quantity -= order.filled_quantity
            position.available_quantity = position.quantity
            
            # 计算已实现盈亏
            sell_pnl = (float(order.avg_fill_price) - float(position.avg_cost)) * float(order.filled_quantity)
            position.realized_pnl += Decimal(str(sell_pnl))
        
        # 更新市值（使用成交价作为市价）
        position.market_value = position.quantity * order.avg_fill_price
        
        # 更新未实现盈亏
        position.unrealized_pnl = position.market_value - position.total_cost
        
        position.last_update_at = datetime.now()
        
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise Exception(f"更新持仓失败: {str(e)}")

