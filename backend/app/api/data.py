"""
数据管理API路由

提供股票数据查询、存储状态检查等接口。
这是一个基础的数据API框架，后续会在Task6中完善。
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.src.storage.parquet_engine import get_parquet_storage
from backend.src.models.basic_models import StockData


router = APIRouter()


class DataSourceInfo(BaseModel):
    """数据源信息模型"""
    name: str
    enabled: bool
    status: str
    last_update: Optional[str] = None


class StorageInfo(BaseModel):
    """存储信息模型"""
    type: str
    path: str
    total_files: int
    total_size_mb: float
    symbols_count: int


class DataStatusResponse(BaseModel):
    """数据状态响应模型"""
    data_sources: List[DataSourceInfo]
    storage_info: List[StorageInfo]
    last_sync: Optional[str] = None


class SymbolListResponse(BaseModel):
    """股票代码列表响应模型"""
    symbols: List[str]
    total_count: int
    data_source: str


class BasicStockBar(BaseModel):
    """基础股票K线数据模型"""
    date: date
    open_price: float = Field(..., description="开盘价")
    close_price: float = Field(..., description="收盘价")
    high_price: float = Field(..., description="最高价")
    low_price: float = Field(..., description="最低价")
    volume: float = Field(..., description="成交量")
    amount: Optional[float] = Field(None, description="成交额")


class StockDataResponse(BaseModel):
    """股票数据响应模型"""
    symbol: str
    name: Optional[str] = None
    bars: List[BasicStockBar]
    total_count: int
    start_date: Optional[date] = None
    end_date: Optional[date] = None


@router.get("/status", response_model=DataStatusResponse, summary="数据状态检查")
async def get_data_status():
    """
    获取数据系统整体状态
    
    返回数据源状态、存储信息、最近同步时间等。
    """
    try:
        # 数据源状态
        data_sources = [
            DataSourceInfo(
                name="AKShare",
                enabled=settings.AKSHARE_ENABLED,
                status="available" if settings.AKSHARE_ENABLED else "disabled"
            )
        ]
        
        # 存储信息
        storage_info = []
        
        # Parquet存储状态
        try:
            parquet_storage = get_parquet_storage(settings.PARQUET_STORAGE_PATH)
            parquet_stats = parquet_storage.get_storage_stats()
            
            storage_info.append(StorageInfo(
                type="parquet",
                path=settings.PARQUET_STORAGE_PATH,
                total_files=parquet_stats.get("total_files", 0),
                total_size_mb=parquet_stats.get("total_size_mb", 0.0),
                symbols_count=parquet_stats.get("symbols_count", 0)
            ))
        except Exception as e:
            storage_info.append(StorageInfo(
                type="parquet",
                path=settings.PARQUET_STORAGE_PATH,
                total_files=0,
                total_size_mb=0.0,
                symbols_count=0
            ))
        
        return DataStatusResponse(
            data_sources=data_sources,
            storage_info=storage_info,
            last_sync=None  # 后续在Task6中实现
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据状态失败: {str(e)}"
        )


@router.get("/symbols", response_model=SymbolListResponse, summary="获取股票代码列表")
async def get_available_symbols(
    source: str = Query("akshare", description="数据源: akshare(真实数据)|storage(本地存储)"),
    limit: int = Query(100, description="返回数量限制", ge=1, le=5000)
):
    """
    获取可用的股票代码列表
    
    支持从AKShare获取真实股票列表，或从本地存储获取已缓存的股票代码。
    """
    try:
        if source == "akshare":
            # 从AKShare获取真实股票列表
            from backend.src.data.akshare_adapter import AKShareAdapter
            adapter = AKShareAdapter()
            stock_list = adapter.get_stock_list()
            
            if not stock_list:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="无法从AKShare获取股票列表"
                )
            
            # 限制返回数量并提取股票代码
            limited_stocks = stock_list[:limit]
            symbols = [stock['symbol'] for stock in limited_stocks]
            
            return SymbolListResponse(
                symbols=symbols,
                total_count=len(symbols),
                data_source="akshare_live"
            )
        else:
            # 从本地存储获取
            parquet_storage = get_parquet_storage(settings.PARQUET_STORAGE_PATH)
            symbols = parquet_storage.list_available_symbols()
            
            return SymbolListResponse(
                symbols=symbols,
                total_count=len(symbols),
                data_source="parquet_storage"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取股票代码列表失败: {str(e)}"
        )


@router.get("/stocks/{symbol}", response_model=StockDataResponse, summary="获取股票历史数据")
async def get_stock_data(
    symbol: str,
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    limit: int = Query(1000, ge=1, le=10000, description="最大返回记录数")
):
    """
    获取指定股票的历史数据
    
    支持按日期范围查询，默认返回最新的1000条记录。
    """
    try:
        parquet_storage = get_parquet_storage(settings.PARQUET_STORAGE_PATH)
        stock_data = parquet_storage.load_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if not stock_data or not stock_data.bars:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到股票 {symbol} 的数据"
            )
        
        # 限制返回记录数
        bars = stock_data.bars[-limit:] if len(stock_data.bars) > limit else stock_data.bars
        
        # 转换数据格式
        basic_bars = []
        for bar in bars:
            basic_bars.append(BasicStockBar(
                date=bar.date,
                open_price=bar.open_price,
                close_price=bar.close_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                volume=bar.volume,
                amount=bar.amount
            ))
        
        # 计算日期范围
        actual_start_date = bars[0].date if bars else None
        actual_end_date = bars[-1].date if bars else None
        
        return StockDataResponse(
            symbol=symbol,
            name=stock_data.name,
            bars=basic_bars,
            total_count=len(basic_bars),
            start_date=actual_start_date,
            end_date=actual_end_date
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取股票数据失败: {str(e)}"
        )


@router.get("/storage/stats", summary="存储统计信息")
async def get_storage_stats():
    """
    获取详细的存储统计信息
    
    返回各种存储引擎的详细统计数据。
    """
    try:
        stats = {}
        
        # Parquet存储统计
        try:
            parquet_storage = get_parquet_storage(settings.PARQUET_STORAGE_PATH)
            parquet_stats = parquet_storage.get_storage_stats()
            stats["parquet"] = parquet_stats
        except Exception as e:
            stats["parquet"] = {"error": str(e)}
        
        # 添加时间戳
        stats["timestamp"] = datetime.utcnow().isoformat()
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取存储统计信息失败: {str(e)}"
        )


@router.post("/test/sample-data", summary="创建测试数据")
async def create_sample_data(symbol: str = "TEST001"):
    """
    创建测试数据 - 仅用于开发和测试
    
    生成样本股票数据用于测试API功能。
    """
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="测试数据接口仅在开发模式下可用"
        )
    
    try:
        from backend.src.models.basic_models import StockDailyBar, StockData
        from datetime import timedelta
        import random
        
        # 生成30天的测试数据
        bars = []
        base_price = 100.0
        current_date = date.today()
        
        for i in range(30):
            # 简单的随机价格生成
            change = random.uniform(-0.05, 0.05)  # ±5%变动
            base_price = base_price * (1 + change)
            
            high_price = base_price * (1 + random.uniform(0, 0.02))
            low_price = base_price * (1 - random.uniform(0, 0.02))
            
            bar = StockDailyBar(
                date=current_date - timedelta(days=29-i),
                open_price=round(base_price, 2),
                close_price=round(base_price, 2),
                high_price=round(high_price, 2),
                low_price=round(low_price, 2),
                volume=random.randint(100000, 1000000),
                amount=random.randint(10000000, 100000000)
            )
            bars.append(bar)
        
        # 创建股票数据
        stock_data = StockData(
            symbol=symbol,
            name=f"测试股票{symbol}",
            bars=bars
        )
        
        # 保存到Parquet存储
        parquet_storage = get_parquet_storage(settings.PARQUET_STORAGE_PATH)
        success = parquet_storage.save_stock_data(stock_data, update_mode="overwrite")
        
        if success:
            return {
                "message": f"成功创建测试数据 {symbol}",
                "symbol": symbol,
                "bars_count": len(bars),
                "date_range": f"{bars[0].date} 到 {bars[-1].date}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="保存测试数据失败"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建测试数据失败: {str(e)}"
        )


@router.get("/akshare/stocks/{symbol}", response_model=StockDataResponse, summary="获取AKShare实时股票数据")
async def get_akshare_stock_data(
    symbol: str,
    days: int = Query(30, ge=1, le=365, description="获取天数"),
    save_to_storage: bool = Query(False, description="是否保存到本地存储")
):
    """
    直接从AKShare获取真实股票数据
    
    每次调用都会从AKShare实时获取数据，确保数据的真实性和时效性。
    可选择将数据保存到本地存储以供后续快速访问。
    """
    try:
        # 验证股票代码格式
        if not symbol.isdigit() or len(symbol) != 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="股票代码格式错误，应为6位数字"
            )
        
        from backend.src.data.akshare_adapter import AKShareAdapter
        from datetime import timedelta
        
        # 计算日期范围
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # 获取AKShare数据
        adapter = AKShareAdapter()
        stock_data = adapter.get_stock_data(symbol, start_date, end_date)
        
        if not stock_data or not stock_data.bars:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"从AKShare未获取到股票 {symbol} 的数据，请检查股票代码是否正确"
            )
        
        # 转换数据格式为API响应格式
        basic_bars = []
        for bar in stock_data.bars:
            basic_bars.append(BasicStockBar(
                date=bar.date,
                open_price=bar.open_price,
                close_price=bar.close_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                volume=bar.volume,
                amount=bar.amount
            ))
        
        # 可选保存到本地存储
        save_result = None
        if save_to_storage:
            try:
                parquet_storage = get_parquet_storage(settings.PARQUET_STORAGE_PATH)
                success = parquet_storage.save_stock_data(stock_data, update_mode="overwrite")
                save_result = {"saved": success}
                if not success:
                    save_result["error"] = "保存到本地存储失败"
            except Exception as e:
                save_result = {"saved": False, "error": str(e)}
        
        response = StockDataResponse(
            symbol=symbol,
            name=stock_data.name,
            bars=basic_bars,
            total_count=len(basic_bars),
            start_date=stock_data.bars[0].date,
            end_date=stock_data.bars[-1].date
        )
        
        # 添加数据源标识
        response_dict = response.dict()
        response_dict["data_source"] = "akshare_live"
        response_dict["fetch_date"] = date.today().isoformat()
        if save_result:
            response_dict["storage_result"] = save_result
            
        return response_dict
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取AKShare股票数据失败: {str(e)}"
        )
