/**
 * WebSocket实时通信服务
 * 提供连接管理、消息分发、自动重连等功能
 */

import { message } from 'antd';

// WebSocket消息类型定义
export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp?: string;
}

// 回测进度消息
export interface BacktestProgressMessage extends WebSocketMessage {
  type: 'backtest_progress';
  data: {
    backtest_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number; // 0-1
    current_step: string;
    estimated_time_remaining?: number;
    error_message?: string;
  };
}

// 系统状态消息
export interface SystemStatusMessage extends WebSocketMessage {
  type: 'system_status';
  data: {
    component: string;
    status: 'online' | 'offline' | 'warning' | 'error';
    message: string;
    details?: any;
  };
}

// 数据同步消息
export interface DataSyncMessage extends WebSocketMessage {
  type: 'data_sync';
  data: {
    source: string;
    action: 'started' | 'completed' | 'failed';
    symbols_updated?: number;
    total_symbols?: number;
    message?: string;
  };
}

// 策略状态更新消息
export interface StrategyStatusUpdateMessage extends WebSocketMessage {
  type: 'strategy_status_update';
  data: {
    strategy_id: string;
    strategy_name: string;
    old_status: string;
    new_status: string;
    performance_data?: any;
    message: string;
  };
}

// 交易信号消息
export interface TradingSignalMessage extends WebSocketMessage {
  type: 'trading_signal';
  data: {
    signal_type: 'entry' | 'exit' | 'position_adjust';
    symbol: string;
    action: 'buy' | 'sell' | 'hold';
    price: number;
    quantity: number;
    strategy_name?: string;
    confidence?: number;
    details?: any;
  };
}

// WebSocket连接状态
export type WebSocketStatus =
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'reconnecting'
  | 'failed';

// 消息处理器类型
export type MessageHandler = (message: WebSocketMessage) => void;

/**
 * WebSocket客户端管理类
 */
class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private status: WebSocketStatus = 'disconnected';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 3000; // 3秒
  private reconnectTimeout: number | null = null;
  private pingInterval: number | null = null;
  private messageHandlers: Map<string, MessageHandler[]> = new Map();
  private statusHandlers: Array<(status: WebSocketStatus) => void> = [];
  private isManualClose = false;

  constructor(url: string = 'ws://localhost:3000/ws') {
    this.url = url;

    // 监听页面可见性变化
    this.setupVisibilityListener();
  }

  /**
   * 连接WebSocket
   */
  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('WebSocket已经连接');
      return;
    }

    try {
      this.setStatus('connecting');
      this.isManualClose = false;

      console.log(`正在连接WebSocket: ${this.url}`);
      this.ws = new WebSocket(this.url);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);
    } catch (error) {
      console.error('WebSocket连接失败:', error);
      this.setStatus('failed');
      this.scheduleReconnect();
    }
  }

  /**
   * 断开WebSocket连接
   */
  disconnect(): void {
    this.isManualClose = true;
    this.clearReconnectTimeout();
    this.clearPingInterval();

    if (this.ws) {
      this.ws.close(1000, '用户主动断开连接');
      this.ws = null;
    }

    this.setStatus('disconnected');
    console.log('WebSocket已断开连接');
  }

  /**
   * 发送消息
   */
  send(message: WebSocketMessage): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket未连接，无法发送消息:', message);
      return false;
    }

    try {
      const messageWithTimestamp = {
        ...message,
        timestamp: new Date().toISOString(),
      };

      this.ws.send(JSON.stringify(messageWithTimestamp));
      console.log('WebSocket消息已发送:', messageWithTimestamp);
      return true;
    } catch (error) {
      console.error('发送WebSocket消息失败:', error);
      return false;
    }
  }

  /**
   * 订阅消息类型
   */
  subscribe(messageType: string, handler: MessageHandler): void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }

    this.messageHandlers.get(messageType)!.push(handler);
    console.log(`已订阅消息类型: ${messageType}`);
  }

  /**
   * 取消订阅消息类型
   */
  unsubscribe(messageType: string, handler?: MessageHandler): void {
    if (!this.messageHandlers.has(messageType)) {
      return;
    }

    if (handler) {
      const handlers = this.messageHandlers.get(messageType)!;
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    } else {
      // 清除该类型的所有处理器
      this.messageHandlers.delete(messageType);
    }

    console.log(`已取消订阅消息类型: ${messageType}`);
  }

  /**
   * 订阅连接状态变化
   */
  onStatusChange(handler: (status: WebSocketStatus) => void): void {
    this.statusHandlers.push(handler);
  }

  /**
   * 获取当前连接状态
   */
  getStatus(): WebSocketStatus {
    return this.status;
  }

  /**
   * 是否已连接
   */
  isConnected(): boolean {
    return this.status === 'connected';
  }

  // 私有方法

  private handleOpen(): void {
    console.log('WebSocket连接已建立');
    this.setStatus('connected');
    this.reconnectAttempts = 0;
    this.clearReconnectTimeout();

    // 开始心跳检测
    this.startPing();

    // 发送连接确认消息
    this.send({
      type: 'client_connected',
      data: {
        user_agent: navigator.userAgent,
        timestamp: Date.now(),
      },
    });

    message.success('实时连接已建立', 2);
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      console.log('收到WebSocket消息:', message);

      // 处理心跳响应
      if (message.type === 'pong') {
        return;
      }

      // 分发消息给订阅者
      this.dispatchMessage(message);
    } catch (error) {
      console.error('解析WebSocket消息失败:', error, event.data);
    }
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket连接关闭:', event.code, event.reason);
    this.clearPingInterval();

    if (!this.isManualClose) {
      this.setStatus('disconnected');
      this.scheduleReconnect();
    } else {
      this.setStatus('disconnected');
    }
  }

  private handleError(error: Event): void {
    console.error('WebSocket错误:', error);
    this.setStatus('failed');

    if (!this.isManualClose) {
      this.scheduleReconnect();
    }
  }

  private setStatus(status: WebSocketStatus): void {
    if (this.status !== status) {
      const oldStatus = this.status;
      this.status = status;

      console.log(`WebSocket状态变化: ${oldStatus} -> ${status}`);

      // 通知状态变化监听器
      this.statusHandlers.forEach(handler => {
        try {
          handler(status);
        } catch (error) {
          console.error('状态处理器执行失败:', error);
        }
      });
    }
  }

  private dispatchMessage(message: WebSocketMessage): void {
    const handlers = this.messageHandlers.get(message.type);
    if (handlers && handlers.length > 0) {
      handlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error(`处理消息类型 ${message.type} 失败:`, error);
        }
      });
    } else {
      console.warn(`未找到消息类型 ${message.type} 的处理器`);
    }
  }

  private scheduleReconnect(): void {
    if (
      this.isManualClose ||
      this.reconnectAttempts >= this.maxReconnectAttempts
    ) {
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('WebSocket重连次数已达上限，停止重连');
        this.setStatus('failed');
        message.error('实时连接失败，请刷新页面重试', 5);
      }
      return;
    }

    this.clearReconnectTimeout();
    this.setStatus('reconnecting');

    const delay =
      this.reconnectInterval * Math.pow(1.5, this.reconnectAttempts);
    console.log(
      `${delay / 1000}秒后尝试重连 (第${this.reconnectAttempts + 1}次)`
    );

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  private clearReconnectTimeout(): void {
    if (this.reconnectTimeout !== null) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  private startPing(): void {
    this.clearPingInterval();

    // 每30秒发送一次心跳
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({
          type: 'ping',
          data: { timestamp: Date.now() },
        });
      }
    }, 30000);
  }

  private clearPingInterval(): void {
    if (this.pingInterval !== null) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  private setupVisibilityListener(): void {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        // 页面隐藏时暂停连接（但不断开）
        console.log('页面已隐藏，暂停WebSocket活动');
        this.clearPingInterval();
      } else {
        // 页面显示时恢复连接
        console.log('页面已显示，恢复WebSocket活动');
        if (this.status === 'connected') {
          this.startPing();
        } else if (!this.isManualClose && this.status !== 'connecting') {
          // 如果连接断开且非用户主动断开，尝试重连
          this.connect();
        }
      }
    });
  }
}

// 全局WebSocket客户端实例
let globalWebSocketClient: WebSocketClient | null = null;

/**
 * 获取全局WebSocket客户端实例
 */
export const getWebSocketClient = (): WebSocketClient => {
  if (!globalWebSocketClient) {
    globalWebSocketClient = new WebSocketClient();
  }
  return globalWebSocketClient;
};

/**
 * WebSocket工具函数
 */
export const websocketUtils = {
  /**
   * 连接WebSocket
   */
  connect: () => {
    const client = getWebSocketClient();
    client.connect();
    return client;
  },

  /**
   * 断开WebSocket连接
   */
  disconnect: () => {
    const client = getWebSocketClient();
    client.disconnect();
  },

  /**
   * 发送消息
   */
  send: (message: WebSocketMessage) => {
    const client = getWebSocketClient();
    return client.send(message);
  },

  /**
   * 订阅消息
   */
  subscribe: (messageType: string, handler: MessageHandler) => {
    const client = getWebSocketClient();
    client.subscribe(messageType, handler);
    return () => client.unsubscribe(messageType, handler); // 返回取消订阅函数
  },

  /**
   * 获取连接状态
   */
  getStatus: () => {
    const client = getWebSocketClient();
    return client.getStatus();
  },

  /**
   * 是否已连接
   */
  isConnected: () => {
    const client = getWebSocketClient();
    return client.isConnected();
  },

  /**
   * 监听状态变化
   */
  onStatusChange: (handler: (status: WebSocketStatus) => void) => {
    const client = getWebSocketClient();
    client.onStatusChange(handler);
  },
};

export default websocketUtils;
