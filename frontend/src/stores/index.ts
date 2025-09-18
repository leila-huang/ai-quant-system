// 统一导出所有状态管理store

// 应用主状态
export {
  useAppStore,
  useSystemStatus,
  useSystemLoading,
  useSystemError,
  useSidebarCollapsed,
  useCurrentPath,
  useTheme,
  useGlobalLoading,
  useLoadingMessage,
  useNotifications,
} from './appStore';

// 数据状态
export {
  useDataStore,
  useStockDataCache,
  useSymbolList,
  useSymbolListLoading,
  useSymbolListError,
  useSelectedSymbols,
  useLoadingSymbols,
  useDataErrors,
} from './dataStore';

// 回测状态
export {
  useBacktestStore,
  useCurrentBacktest,
  useBacktestHistory,
  useBacktestForm,
  useFormValid,
  useFormErrors,
  useSupportedStrategies,
  useIsRunning,
  useIsLoadingHistory,
  useShowResults,
  useSelectedBacktestIds,
} from './backtestStore';

// WebSocket状态
export {
  useWebSocketStore,
  websocketSelectors,
  useWebSocketConnection,
  useBacktestProgress,
  useSystemStatus as useWebSocketSystemStatus, // 重命名以避免冲突
  useWebSocketNotifications,
} from './websocketStore';
