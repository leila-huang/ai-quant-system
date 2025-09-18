/**
 * WebSocket状态管理Store
 * 管理WebSocket连接状态、消息分发、实时数据更新等
 */

import { useEffect, useCallback, useMemo } from 'react';
import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { websocketUtils, getWebSocketClient } from '../services/websocket';
import type {
  WebSocketStatus,
  WebSocketMessage,
  BacktestProgressMessage,
  SystemStatusMessage,
  DataSyncMessage,
} from '../services/websocket';

interface WebSocketState {
  // 连接状态
  status: WebSocketStatus;
  isConnected: boolean;
  lastConnectTime?: string;
  lastDisconnectTime?: string;
  reconnectAttempts: number;

  // 消息统计
  messageCount: number;
  lastMessageTime?: string;

  // 回测进度消息
  backtestProgresses: Map<string, BacktestProgressMessage['data']>;

  // 系统状态消息
  systemStatuses: Map<string, SystemStatusMessage['data']>;

  // 数据同步状态
  dataSyncStatuses: Map<string, DataSyncMessage['data']>;

  // 实时通知队列
  notifications: WebSocketMessage[];
  maxNotifications: number;
}

interface WebSocketActions {
  // 连接控制
  connect: () => void;
  disconnect: () => void;

  // 状态更新
  setStatus: (status: WebSocketStatus) => void;
  incrementMessageCount: () => void;

  // 消息处理
  handleBacktestProgress: (message: BacktestProgressMessage) => void;
  handleSystemStatus: (message: SystemStatusMessage) => void;
  handleDataSync: (message: DataSyncMessage) => void;

  // 通知管理
  addNotification: (message: WebSocketMessage) => void;
  removeNotification: (index: number) => void;
  clearNotifications: () => void;

  // 状态查询
  getBacktestProgress: (
    backtestId: string
  ) => BacktestProgressMessage['data'] | undefined;
  getSystemStatus: (
    component: string
  ) => SystemStatusMessage['data'] | undefined;
  getDataSyncStatus: (source: string) => DataSyncMessage['data'] | undefined;

  // 重置状态
  reset: () => void;

  // 内部方法
  setupMessageHandlers: (client: any) => void;
}

type WebSocketStore = WebSocketState & WebSocketActions;

// 初始状态
const initialState: WebSocketState = {
  status: 'disconnected',
  isConnected: false,
  reconnectAttempts: 0,
  messageCount: 0,
  backtestProgresses: new Map(),
  systemStatuses: new Map(),
  dataSyncStatuses: new Map(),
  notifications: [],
  maxNotifications: 50,
};

/**
 * WebSocket状态管理Store
 */
export const useWebSocketStore = create<WebSocketStore>()(
  devtools(
    subscribeWithSelector(
      (set, get): WebSocketStore => ({
        ...initialState,

        // 连接控制
        connect: () => {
          const state = get();
          // 避免重复连接
          if (state.status === 'connected' || state.status === 'connecting') {
            return;
          }

          const client = getWebSocketClient();

          // 订阅状态变化
          client.onStatusChange((status: WebSocketStatus) => {
            set(state => ({
              status,
              isConnected: status === 'connected',
              lastConnectTime:
                status === 'connected'
                  ? new Date().toISOString()
                  : state.lastConnectTime,
              lastDisconnectTime:
                status === 'disconnected'
                  ? new Date().toISOString()
                  : state.lastDisconnectTime,
            }));
          });

          // 订阅各类消息
          state.setupMessageHandlers(client);

          // 开始连接
          client.connect();
        },

        disconnect: () => {
          websocketUtils.disconnect();
        },

        // 状态更新
        setStatus: status =>
          set({ status, isConnected: status === 'connected' }),

        incrementMessageCount: () =>
          set(state => ({
            messageCount: state.messageCount + 1,
            lastMessageTime: new Date().toISOString(),
          })),

        // 消息处理
        handleBacktestProgress: message => {
          const { backtestProgresses } = get();
          backtestProgresses.set(message.data.backtest_id, message.data);

          set({ backtestProgresses: new Map(backtestProgresses) });
          get().addNotification(message);
          get().incrementMessageCount();
        },

        handleSystemStatus: message => {
          const { systemStatuses } = get();
          systemStatuses.set(message.data.component, message.data);

          set({ systemStatuses: new Map(systemStatuses) });
          get().addNotification(message);
          get().incrementMessageCount();
        },

        handleDataSync: message => {
          const { dataSyncStatuses } = get();
          dataSyncStatuses.set(message.data.source, message.data);

          set({ dataSyncStatuses: new Map(dataSyncStatuses) });
          get().addNotification(message);
          get().incrementMessageCount();
        },

        // 通知管理
        addNotification: message =>
          set(state => {
            const newNotifications = [...state.notifications, message];

            // 限制通知数量
            if (newNotifications.length > state.maxNotifications) {
              newNotifications.splice(
                0,
                newNotifications.length - state.maxNotifications
              );
            }

            return { notifications: newNotifications };
          }),

        removeNotification: index =>
          set(state => ({
            notifications: state.notifications.filter((_, i) => i !== index),
          })),

        clearNotifications: () => set({ notifications: [] }),

        // 状态查询
        getBacktestProgress: backtestId => {
          return get().backtestProgresses.get(backtestId);
        },

        getSystemStatus: component => {
          return get().systemStatuses.get(component);
        },

        getDataSyncStatus: source => {
          return get().dataSyncStatuses.get(source);
        },

        // 重置状态
        reset: () => set(initialState),

        // 内部方法：设置消息处理器
        setupMessageHandlers: (client: any) => {
          const store = get();

          // 回测进度消息
          client.subscribe('backtest_progress', (message: any) => {
            store.handleBacktestProgress(message as BacktestProgressMessage);
          });

          // 系统状态消息
          client.subscribe('system_status', (message: any) => {
            store.handleSystemStatus(message as SystemStatusMessage);
          });

          // 数据同步消息
          client.subscribe('data_sync', (message: any) => {
            store.handleDataSync(message as DataSyncMessage);
          });

          // 通用消息处理
          client.subscribe('notification', (message: any) => {
            store.addNotification(message);
            store.incrementMessageCount();
          });
        },
      })
    ),
    {
      name: 'websocket-store',
      partialize: (state: WebSocketStore) => ({
        // 只持久化部分状态，不包括连接状态和实时数据
        messageCount: state.messageCount,
        maxNotifications: state.maxNotifications,
      }),
    }
  )
);

/**
 * WebSocket状态选择器
 */
export const websocketSelectors = {
  // 连接状态
  isConnected: (state: WebSocketStore) => state.isConnected,
  status: (state: WebSocketStore) => state.status,
  lastConnectTime: (state: WebSocketStore) => state.lastConnectTime,

  // 消息统计
  messageCount: (state: WebSocketStore) => state.messageCount,
  lastMessageTime: (state: WebSocketStore) => state.lastMessageTime,

  // 回测进度
  allBacktestProgresses: (state: WebSocketStore) =>
    Array.from(state.backtestProgresses.entries()),
  backtestProgress: (backtestId: string) => (state: WebSocketStore) =>
    state.backtestProgresses.get(backtestId),

  // 系统状态
  allSystemStatuses: (state: WebSocketStore) =>
    Array.from(state.systemStatuses.entries()),
  systemStatus: (component: string) => (state: WebSocketStore) =>
    state.systemStatuses.get(component),

  // 数据同步状态
  allDataSyncStatuses: (state: WebSocketStore) =>
    Array.from(state.dataSyncStatuses.entries()),
  dataSyncStatus: (source: string) => (state: WebSocketStore) =>
    state.dataSyncStatuses.get(source),

  // 通知
  notifications: (state: WebSocketStore) => state.notifications,
  latestNotifications: (count: number) => (state: WebSocketStore) =>
    state.notifications.slice(-count),
};

/**
 * WebSocket Hook - 自动连接和清理
 */
export const useWebSocketConnection = (autoConnect = true) => {
  const status = useWebSocketStore(state => state.status);
  const isConnected = useWebSocketStore(state => state.isConnected);
  const connect = useWebSocketStore(state => state.connect);
  const disconnect = useWebSocketStore(state => state.disconnect);

  // 挂载时根据autoConnect进行连接（仅执行一次，不随status变化清理）
  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    // 仅在卸载时断开，避免依赖变化触发disconnect→status变化→re-render循环
    return () => {
      disconnect();
    };
  }, [autoConnect]);

  return { connect, disconnect, status, isConnected };
};

/**
 * 回测进度Hook
 */
export const useBacktestProgress = (backtestId?: string) => {
  const backtestProgresses = useWebSocketStore(
    websocketSelectors.allBacktestProgresses
  );
  const getProgress = useWebSocketStore(state => state.getBacktestProgress);

  if (backtestId) {
    return getProgress(backtestId);
  }

  return backtestProgresses;
};

/**
 * 系统状态Hook
 */
export const useSystemStatus = (component?: string) => {
  const systemStatuses = useWebSocketStore(
    websocketSelectors.allSystemStatuses
  );
  const getStatus = useWebSocketStore(state => state.getSystemStatus);

  if (component) {
    return getStatus(component);
  }

  return systemStatuses;
};

/**
 * 实时通知Hook
 */
export const useWebSocketNotifications = (maxCount = 10) => {
  const allNotifications = useWebSocketStore(state => state.notifications);
  const notifications = useMemo(
    () => allNotifications.slice(-maxCount),
    [allNotifications, maxCount]
  );

  // 使用useCallback确保函数引用稳定
  const removeNotification = useCallback(
    (index: number) => useWebSocketStore.getState().removeNotification(index),
    []
  );

  const clearNotifications = useCallback(
    () => useWebSocketStore.getState().clearNotifications(),
    []
  );

  return {
    notifications,
    removeNotification,
    clearNotifications,
  };
};
