import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { StockDataResponse, SymbolListResponse } from '@/types/api';

// 股票数据缓存项
interface StockDataCache {
  [symbol: string]: {
    data: StockDataResponse;
    timestamp: number;
    expiry: number; // 缓存过期时间
  };
}

// 数据状态接口
interface DataState {
  // 股票数据缓存
  stockDataCache: StockDataCache;

  // 股票列表
  symbolList: SymbolListResponse | null;
  symbolListLoading: boolean;
  symbolListError: string | null;

  // 当前选中的股票
  selectedSymbols: string[];

  // 数据加载状态
  loadingSymbols: Set<string>;
  errors: Record<string, string>;

  // 缓存配置
  cacheExpiration: number; // 缓存有效期（毫秒）

  // Actions
  setStockData: (symbol: string, data: StockDataResponse) => void;
  getStockData: (symbol: string) => StockDataResponse | null;
  isStockDataFresh: (symbol: string) => boolean;
  clearStockDataCache: () => void;
  removeStockData: (symbol: string) => void;

  setSymbolList: (list: SymbolListResponse) => void;
  setSymbolListLoading: (loading: boolean) => void;
  setSymbolListError: (error: string | null) => void;

  setSelectedSymbols: (symbols: string[]) => void;
  addSelectedSymbol: (symbol: string) => void;
  removeSelectedSymbol: (symbol: string) => void;

  setSymbolLoading: (symbol: string, loading: boolean) => void;
  setSymbolError: (symbol: string, error: string | null) => void;
  clearSymbolError: (symbol: string) => void;

  setCacheExpiration: (expiration: number) => void;
}

// 默认缓存有效期：5分钟
const DEFAULT_CACHE_EXPIRATION = 5 * 60 * 1000;

// 创建数据状态store
export const useDataStore = create<DataState>()(
  devtools(
    (set): DataState => ({
      // 初始状态
      stockDataCache: {},

      symbolList: null,
      symbolListLoading: false,
      symbolListError: null,

      selectedSymbols: [],

      loadingSymbols: new Set(),
      errors: {},

      cacheExpiration: DEFAULT_CACHE_EXPIRATION,

      // Actions
      setStockData: (symbol, data) =>
        set(
          state => ({
            stockDataCache: {
              ...state.stockDataCache,
              [symbol]: {
                data,
                timestamp: Date.now(),
                expiry: Date.now() + state.cacheExpiration,
              },
            },
          }),
          false,
          `setStockData:${symbol}`
        ),

      getStockData: symbol => {
        const state = useDataStore.getState();
        const cached = state.stockDataCache[symbol];

        if (!cached) return null;

        // 检查缓存是否过期
        if (Date.now() > cached.expiry) {
          // 异步清理过期缓存
          set(
            prevState => {
              const newCache = { ...prevState.stockDataCache };
              delete newCache[symbol];
              return { stockDataCache: newCache };
            },
            false,
            `expireCache:${symbol}`
          );
          return null;
        }

        return cached.data;
      },

      isStockDataFresh: (symbol: string) => {
        const state = useDataStore.getState();
        const cached = state.stockDataCache[symbol];
        return cached && Date.now() <= cached.expiry;
      },

      clearStockDataCache: () =>
        set({ stockDataCache: {} }, false, 'clearStockDataCache'),

      removeStockData: (symbol: string) =>
        set(
          state => {
            const newCache = { ...state.stockDataCache };
            delete newCache[symbol];
            return { stockDataCache: newCache };
          },
          false,
          `removeStockData:${symbol}`
        ),

      setSymbolList: (list: SymbolListResponse) =>
        set(
          { symbolList: list, symbolListError: null },
          false,
          'setSymbolList'
        ),

      setSymbolListLoading: (loading: boolean) =>
        set({ symbolListLoading: loading }, false, 'setSymbolListLoading'),

      setSymbolListError: (error: string | null) =>
        set({ symbolListError: error }, false, 'setSymbolListError'),

      setSelectedSymbols: (symbols: string[]) =>
        set({ selectedSymbols: symbols }, false, 'setSelectedSymbols'),

      addSelectedSymbol: (symbol: string) =>
        set(
          state => ({
            selectedSymbols: state.selectedSymbols.includes(symbol)
              ? state.selectedSymbols
              : [...state.selectedSymbols, symbol],
          }),
          false,
          `addSelectedSymbol:${symbol}`
        ),

      removeSelectedSymbol: (symbol: string) =>
        set(
          state => ({
            selectedSymbols: state.selectedSymbols.filter(s => s !== symbol),
          }),
          false,
          `removeSelectedSymbol:${symbol}`
        ),

      setSymbolLoading: (symbol: string, loading: boolean) =>
        set(
          state => {
            const newLoadingSymbols = new Set(state.loadingSymbols);
            if (loading) {
              newLoadingSymbols.add(symbol);
            } else {
              newLoadingSymbols.delete(symbol);
            }
            return { loadingSymbols: newLoadingSymbols };
          },
          false,
          `setSymbolLoading:${symbol}`
        ),

      setSymbolError: (symbol: string, error: string | null) =>
        set(
          state => {
            if (error) {
              return { errors: { ...state.errors, [symbol]: error } };
            } else {
              const newErrors = { ...state.errors };
              delete newErrors[symbol];
              return { errors: newErrors };
            }
          },
          false,
          `setSymbolError:${symbol}`
        ),

      clearSymbolError: (symbol: string) =>
        set(
          state => {
            const newErrors = { ...state.errors };
            delete newErrors[symbol];
            return { errors: newErrors };
          },
          false,
          `clearSymbolError:${symbol}`
        ),

      setCacheExpiration: (expiration: number) =>
        set({ cacheExpiration: expiration }, false, 'setCacheExpiration'),
    }),
    {
      name: 'data-store',
    }
  )
);

// 导出常用的状态选择器
export const useStockDataCache = () =>
  useDataStore(state => state.stockDataCache);
export const useSymbolList = () => useDataStore(state => state.symbolList);
export const useSymbolListLoading = () =>
  useDataStore(state => state.symbolListLoading);
export const useSymbolListError = () =>
  useDataStore(state => state.symbolListError);
export const useSelectedSymbols = () =>
  useDataStore(state => state.selectedSymbols);
export const useLoadingSymbols = () =>
  useDataStore(state => state.loadingSymbols);
export const useDataErrors = () => useDataStore(state => state.errors);

export default useDataStore;
