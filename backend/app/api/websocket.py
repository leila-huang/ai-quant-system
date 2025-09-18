"""
WebSocket实时通信接口

提供实时数据推送、系统状态通知、回测进度推送等WebSocket功能。
"""

import json
import logging
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 活动连接列表
        self.active_connections: List[WebSocket] = []
        # 连接元数据
        self.connection_info: Dict[WebSocket, dict] = {}
    
    async def connect(self, websocket: WebSocket, client_info: dict = None):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = client_info or {}
        
        logger.info(f"WebSocket连接已建立，当前连接数: {len(self.active_connections)}")
        
        # 发送欢迎消息
        await self.send_personal_message({
            "type": "connection_established",
            "data": {
                "message": "WebSocket连接已建立",
                "timestamp": datetime.now().isoformat(),
                "connection_count": len(self.active_connections)
            }
        }, websocket)
    
    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_info:
            del self.connection_info[websocket]
        
        logger.info(f"WebSocket连接已断开，当前连接数: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """发送个人消息"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送个人消息失败: {e}")
            # 如果发送失败，可能连接已断开，移除连接
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """广播消息给所有连接"""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message, ensure_ascii=False)
        disconnected_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected_connections.append(connection)
        
        # 清理断开的连接
        for connection in disconnected_connections:
            self.disconnect(connection)
        
        if disconnected_connections:
            logger.info(f"清理了 {len(disconnected_connections)} 个断开的连接")
    
    async def send_system_status(self, component: str, status: str, message: str, details: dict = None):
        """发送系统状态消息"""
        await self.broadcast({
            "type": "system_status",
            "data": {
                "component": component,
                "status": status,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def send_backtest_progress(self, backtest_id: str, status: str, progress: float, 
                                   current_step: str, estimated_time_remaining: int = None,
                                   error_message: str = None):
        """发送回测进度消息"""
        await self.broadcast({
            "type": "backtest_progress",
            "data": {
                "backtest_id": backtest_id,
                "status": status,
                "progress": progress,
                "current_step": current_step,
                "estimated_time_remaining": estimated_time_remaining,
                "error_message": error_message,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def send_data_sync_update(self, source: str, action: str, symbols_updated: int = None,
                                  total_symbols: int = None, message: str = None):
        """发送数据同步更新消息"""
        await self.broadcast({
            "type": "data_sync",
            "data": {
                "source": source,
                "action": action,
                "symbols_updated": symbols_updated,
                "total_symbols": total_symbols,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def send_strategy_status_update(self, strategy_id: str, strategy_name: str, 
                                        old_status: str, new_status: str, 
                                        performance_data: dict = None, message: str = None):
        """发送策略状态更新消息"""
        await self.broadcast({
            "type": "strategy_status_update",
            "data": {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "old_status": old_status,
                "new_status": new_status,
                "performance_data": performance_data or {},
                "message": message or f"策略 {strategy_name} 状态从 {old_status} 变更为 {new_status}",
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def send_trading_signal(self, signal_type: str, symbol: str, action: str,
                                price: float, quantity: int, strategy_name: str = None,
                                confidence: float = None, details: dict = None):
        """发送交易信号消息"""
        await self.broadcast({
            "type": "trading_signal",
            "data": {
                "signal_type": signal_type,  # "entry", "exit", "position_adjust"
                "symbol": symbol,
                "action": action,  # "buy", "sell", "hold"
                "price": price,
                "quantity": quantity,
                "strategy_name": strategy_name,
                "confidence": confidence,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        })
    
    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(self.active_connections)
    
    def get_connection_info(self) -> List[dict]:
        """获取所有连接信息"""
        return [
            {
                "connection_id": id(ws),
                "info": self.connection_info.get(ws, {}),
                "connected_at": self.connection_info.get(ws, {}).get("connected_at")
            }
            for ws in self.active_connections
        ]


# 全局连接管理器实例
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket主端点"""
    client_host = websocket.client.host if websocket.client else "unknown"
    client_info = {
        "client_host": client_host,
        "connected_at": datetime.now().isoformat(),
        "user_agent": websocket.headers.get("user-agent", "unknown")
    }
    
    await manager.connect(websocket, client_info)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_client_message(message, websocket)
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "data": {
                        "message": "消息格式错误，请发送有效的JSON",
                        "timestamp": datetime.now().isoformat()
                    }
                }, websocket)
            except Exception as e:
                logger.error(f"处理客户端消息失败: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "data": {
                        "message": f"处理消息时发生错误: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("客户端主动断开连接")
    except Exception as e:
        logger.error(f"WebSocket连接异常: {e}")
        manager.disconnect(websocket)


async def handle_client_message(message: dict, websocket: WebSocket):
    """处理客户端消息"""
    message_type = message.get("type", "unknown")
    message_data = message.get("data", {})
    
    logger.info(f"收到客户端消息: {message_type}")
    
    if message_type == "ping":
        # 心跳检测
        await manager.send_personal_message({
            "type": "pong",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "server_time": message_data.get("timestamp")
            }
        }, websocket)
    
    elif message_type == "client_connected":
        # 客户端连接确认
        await manager.send_personal_message({
            "type": "server_ack",
            "data": {
                "message": "服务器已确认连接",
                "client_info": message_data,
                "timestamp": datetime.now().isoformat()
            }
        }, websocket)
    
    elif message_type == "get_status":
        # 获取系统状态
        await manager.send_personal_message({
            "type": "system_status",
            "data": {
                "component": "websocket_server",
                "status": "online",
                "message": "WebSocket服务正常运行",
                "details": {
                    "active_connections": manager.get_connection_count(),
                    "server_time": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
        }, websocket)
    
    else:
        # 未知消息类型
        await manager.send_personal_message({
            "type": "error",
            "data": {
                "message": f"未知的消息类型: {message_type}",
                "timestamp": datetime.now().isoformat()
            }
        }, websocket)


# 工具函数，供其他模块使用
async def broadcast_system_status(component: str, status: str, message: str, details: dict = None):
    """广播系统状态（供外部调用）"""
    await manager.send_system_status(component, status, message, details)


async def broadcast_backtest_progress(backtest_id: str, status: str, progress: float,
                                    current_step: str, estimated_time_remaining: int = None,
                                    error_message: str = None):
    """广播回测进度（供外部调用）"""
    await manager.send_backtest_progress(
        backtest_id, status, progress, current_step, 
        estimated_time_remaining, error_message
    )


async def broadcast_data_sync_update(source: str, action: str, symbols_updated: int = None,
                                   total_symbols: int = None, message: str = None):
    """广播数据同步更新（供外部调用）"""
    await manager.send_data_sync_update(
        source, action, symbols_updated, total_symbols, message
    )


async def broadcast_strategy_status_update(strategy_id: str, strategy_name: str,
                                         old_status: str, new_status: str,
                                         performance_data: dict = None, message: str = None):
    """广播策略状态更新（供外部调用）"""
    await manager.send_strategy_status_update(
        strategy_id, strategy_name, old_status, new_status, performance_data, message
    )


async def broadcast_trading_signal(signal_type: str, symbol: str, action: str,
                                 price: float, quantity: int, strategy_name: str = None,
                                 confidence: float = None, details: dict = None):
    """广播交易信号（供外部调用）"""
    await manager.send_trading_signal(
        signal_type, symbol, action, price, quantity, strategy_name, confidence, details
    )


def get_websocket_stats() -> dict:
    """获取WebSocket统计信息"""
    return {
        "active_connections": manager.get_connection_count(),
        "connection_info": manager.get_connection_info(),
        "server_time": datetime.now().isoformat()
    }
