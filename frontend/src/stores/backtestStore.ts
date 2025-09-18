import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
  BacktestRequest,
  BacktestResponse,
  BacktestStatusResponse,
  StrategyConfig,
} from '@/types/api';

// 回测任务状态
interface BacktestTask {
  id: string;
  request: BacktestRequest;
  response?: BacktestResponse;
  status?: BacktestStatusResponse;
  error?: string;
  results?: Record<string, any>;
}

// 回测状态接口
interface BacktestState {
  // 当前回测任务
  currentBacktest: BacktestTask | null;

  // 回测历史
  backtestHistory: BacktestTask[];

  // 回测表单状态
  backtestForm: Partial<BacktestRequest>;
  formValid: boolean;
  formErrors: Record<string, string>;

  // 支持的策略配置
  supportedStrategies: Record<string, any> | null;

  // UI状态
  isRunning: boolean;
  isLoadingHistory: boolean;
  showResults: boolean;
  selectedBacktestIds: string[];

  // Actions
  setCurrentBacktest: (backtest: BacktestTask | null) => void;
  updateBacktestStatus: (
    backtestId: string,
    status: BacktestStatusResponse
  ) => void;
  updateBacktestResults: (
    backtestId: string,
    results: Record<string, any>
  ) => void;
  addBacktestError: (backtestId: string, error: string) => void;

  setBacktestHistory: (history: BacktestTask[]) => void;
  addToHistory: (backtest: BacktestTask) => void;
  removeFromHistory: (backtestId: string) => void;
  clearHistory: () => void;

  updateBacktestForm: (updates: Partial<BacktestRequest>) => void;
  setFormValid: (valid: boolean) => void;
  setFormErrors: (errors: Record<string, string>) => void;
  resetBacktestForm: () => void;

  setSupportedStrategies: (strategies: Record<string, any>) => void;

  setIsRunning: (running: boolean) => void;
  setIsLoadingHistory: (loading: boolean) => void;
  setShowResults: (show: boolean) => void;
  setSelectedBacktestIds: (ids: string[]) => void;
  toggleBacktestSelection: (id: string) => void;
}

// 默认回测表单配置
const getDefaultBacktestForm = (): Partial<BacktestRequest> => ({
  backtest_name: '',
  strategy_config: {
    strategy_type: 'ma_crossover',
    strategy_name: '双均线策略',
    parameters: {
      fast_period: 5,
      slow_period: 20,
      stop_loss: 0.05,
    },
  } as StrategyConfig,
  universe: ['000001', '000002'], // 默认股票池
  start_date: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000)
    .toISOString()
    .split('T')[0], // 1年前
  end_date: new Date().toISOString().split('T')[0], // 今天
  initial_capital: 1000000,
  commission: 0.0003,
  enable_cost_model: true,
  broker_type: 'standard',
  enable_constraints: true,
  price_limit_enabled: true,
  t_plus_1_enabled: true,
  rebalance_frequency: 'daily',
  benchmark_symbol: '000300.SH',
  async_execution: false,
});

// 创建回测状态store
export const useBacktestStore = create<BacktestState>()(
  devtools(
    set => ({
      // 初始状态
      currentBacktest: null,
      backtestHistory: [],

      backtestForm: getDefaultBacktestForm(),
      formValid: false,
      formErrors: {},

      supportedStrategies: null,

      isRunning: false,
      isLoadingHistory: false,
      showResults: false,
      selectedBacktestIds: [],

      // Actions
      setCurrentBacktest: backtest =>
        set({ currentBacktest: backtest }, false, 'setCurrentBacktest'),

      updateBacktestStatus: (backtestId, status) =>
        set(
          state => {
            // 更新当前回测状态
            const updatedCurrent =
              state.currentBacktest?.id === backtestId
                ? { ...state.currentBacktest, status }
                : state.currentBacktest;

            // 更新历史记录中的状态
            const updatedHistory = state.backtestHistory.map(bt =>
              bt.id === backtestId ? { ...bt, status } : bt
            );

            return {
              currentBacktest: updatedCurrent,
              backtestHistory: updatedHistory,
            };
          },
          false,
          `updateBacktestStatus:${backtestId}`
        ),

      updateBacktestResults: (backtestId, results) =>
        set(
          state => {
            const updatedCurrent =
              state.currentBacktest?.id === backtestId
                ? { ...state.currentBacktest, results }
                : state.currentBacktest;

            const updatedHistory = state.backtestHistory.map(bt =>
              bt.id === backtestId ? { ...bt, results } : bt
            );

            return {
              currentBacktest: updatedCurrent,
              backtestHistory: updatedHistory,
            };
          },
          false,
          `updateBacktestResults:${backtestId}`
        ),

      addBacktestError: (backtestId, error) =>
        set(
          state => {
            const updatedCurrent =
              state.currentBacktest?.id === backtestId
                ? { ...state.currentBacktest, error }
                : state.currentBacktest;

            const updatedHistory = state.backtestHistory.map(bt =>
              bt.id === backtestId ? { ...bt, error } : bt
            );

            return {
              currentBacktest: updatedCurrent,
              backtestHistory: updatedHistory,
            };
          },
          false,
          `addBacktestError:${backtestId}`
        ),

      setBacktestHistory: history =>
        set({ backtestHistory: history }, false, 'setBacktestHistory'),

      addToHistory: backtest =>
        set(
          state => ({
            backtestHistory: [backtest, ...state.backtestHistory],
          }),
          false,
          'addToHistory'
        ),

      removeFromHistory: backtestId =>
        set(
          state => ({
            backtestHistory: state.backtestHistory.filter(
              bt => bt.id !== backtestId
            ),
          }),
          false,
          `removeFromHistory:${backtestId}`
        ),

      clearHistory: () => set({ backtestHistory: [] }, false, 'clearHistory'),

      updateBacktestForm: updates =>
        set(
          state => ({
            backtestForm: { ...state.backtestForm, ...updates },
          }),
          false,
          'updateBacktestForm'
        ),

      setFormValid: valid => set({ formValid: valid }, false, 'setFormValid'),

      setFormErrors: errors =>
        set({ formErrors: errors }, false, 'setFormErrors'),

      resetBacktestForm: () =>
        set(
          {
            backtestForm: getDefaultBacktestForm(),
            formValid: false,
            formErrors: {},
          },
          false,
          'resetBacktestForm'
        ),

      setSupportedStrategies: strategies =>
        set(
          { supportedStrategies: strategies },
          false,
          'setSupportedStrategies'
        ),

      setIsRunning: running =>
        set({ isRunning: running }, false, 'setIsRunning'),

      setIsLoadingHistory: loading =>
        set({ isLoadingHistory: loading }, false, 'setIsLoadingHistory'),

      setShowResults: show =>
        set({ showResults: show }, false, 'setShowResults'),

      setSelectedBacktestIds: ids =>
        set({ selectedBacktestIds: ids }, false, 'setSelectedBacktestIds'),

      toggleBacktestSelection: id =>
        set(
          state => ({
            selectedBacktestIds: state.selectedBacktestIds.includes(id)
              ? state.selectedBacktestIds.filter(
                  selectedId => selectedId !== id
                )
              : [...state.selectedBacktestIds, id],
          }),
          false,
          `toggleBacktestSelection:${id}`
        ),
    }),
    {
      name: 'backtest-store',
    }
  )
);

// 导出常用的状态选择器
export const useCurrentBacktest = () =>
  useBacktestStore(state => state.currentBacktest);
export const useBacktestHistory = () =>
  useBacktestStore(state => state.backtestHistory);
export const useBacktestForm = () =>
  useBacktestStore(state => state.backtestForm);
export const useFormValid = () => useBacktestStore(state => state.formValid);
export const useFormErrors = () => useBacktestStore(state => state.formErrors);
export const useSupportedStrategies = () =>
  useBacktestStore(state => state.supportedStrategies);
export const useIsRunning = () => useBacktestStore(state => state.isRunning);
export const useIsLoadingHistory = () =>
  useBacktestStore(state => state.isLoadingHistory);
export const useShowResults = () =>
  useBacktestStore(state => state.showResults);
export const useSelectedBacktestIds = () =>
  useBacktestStore(state => state.selectedBacktestIds);

export default useBacktestStore;

