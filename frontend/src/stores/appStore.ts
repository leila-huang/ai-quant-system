import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { DataStatusResponse } from '@/types/api';

// 应用全局状态接口
interface AppState {
  // 系统状态
  systemStatus: DataStatusResponse | null;
  isSystemLoading: boolean;
  systemError: string | null;

  // 用户界面状态
  sidebarCollapsed: boolean;
  currentPath: string;
  theme: 'light' | 'dark';

  // 全局加载状态
  globalLoading: boolean;
  loadingMessage: string;

  // 错误和通知
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    title: string;
    message: string;
    timestamp: number;
  }>;

  // Actions
  setSystemStatus: (status: DataStatusResponse) => void;
  setSystemLoading: (loading: boolean) => void;
  setSystemError: (error: string | null) => void;

  setSidebarCollapsed: (collapsed: boolean) => void;
  setCurrentPath: (path: string) => void;
  setTheme: (theme: 'light' | 'dark') => void;

  setGlobalLoading: (loading: boolean, message?: string) => void;

  addNotification: (
    notification: Omit<AppState['notifications'][0], 'id' | 'timestamp'>
  ) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

// 创建应用状态store
export const useAppStore = create<AppState>()(
  devtools(
    persist(
      set => ({
        // 初始状态
        systemStatus: null,
        isSystemLoading: false,
        systemError: null,

        sidebarCollapsed: false,
        currentPath: '/dashboard',
        theme: 'light',

        globalLoading: false,
        loadingMessage: '',

        notifications: [],

        // Actions
        setSystemStatus: status =>
          set(
            { systemStatus: status, systemError: null },
            false,
            'setSystemStatus'
          ),

        setSystemLoading: loading =>
          set({ isSystemLoading: loading }, false, 'setSystemLoading'),

        setSystemError: error =>
          set({ systemError: error }, false, 'setSystemError'),

        setSidebarCollapsed: collapsed =>
          set({ sidebarCollapsed: collapsed }, false, 'setSidebarCollapsed'),

        setCurrentPath: path =>
          set({ currentPath: path }, false, 'setCurrentPath'),

        setTheme: theme => set({ theme }, false, 'setTheme'),

        setGlobalLoading: (loading, message = '') =>
          set(
            { globalLoading: loading, loadingMessage: message },
            false,
            'setGlobalLoading'
          ),

        addNotification: notification =>
          set(
            state => ({
              notifications: [
                ...state.notifications,
                {
                  ...notification,
                  id: `notification_${Date.now()}_${Math.random()}`,
                  timestamp: Date.now(),
                },
              ],
            }),
            false,
            'addNotification'
          ),

        removeNotification: id =>
          set(
            state => ({
              notifications: state.notifications.filter(n => n.id !== id),
            }),
            false,
            'removeNotification'
          ),

        clearNotifications: () =>
          set({ notifications: [] }, false, 'clearNotifications'),
      }),
      {
        name: 'ai-quant-app-store',
        // 持久化配置：只保存用户偏好，不保存临时状态
        partialize: state => ({
          sidebarCollapsed: state.sidebarCollapsed,
          theme: state.theme,
        }),
      }
    ),
    {
      name: 'app-store',
    }
  )
);

// 导出store hook和状态选择器
export const useSystemStatus = () => useAppStore(state => state.systemStatus);
export const useSystemLoading = () =>
  useAppStore(state => state.isSystemLoading);
export const useSystemError = () => useAppStore(state => state.systemError);

export const useSidebarCollapsed = () =>
  useAppStore(state => state.sidebarCollapsed);
export const useCurrentPath = () => useAppStore(state => state.currentPath);
export const useTheme = () => useAppStore(state => state.theme);

export const useGlobalLoading = () => useAppStore(state => state.globalLoading);
export const useLoadingMessage = () =>
  useAppStore(state => state.loadingMessage);

export const useNotifications = () => useAppStore(state => state.notifications);

export default useAppStore;

