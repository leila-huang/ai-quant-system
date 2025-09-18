import { apiRequest } from './api';
import type {
  DataStatusResponse,
  SymbolListResponse,
  StockDataResponse,
  GetStockDataParams,
  GetSymbolsParams,
} from '@/types/api';

/**
 * 数据管理相关API服务
 */
export const dataApi = {
  /**
   * 获取数据系统整体状态
   */
  getStatus: (): Promise<DataStatusResponse> => {
    return apiRequest.get('/data/status');
  },

  /**
   * 获取存储统计信息
   */
  getStorageStats: (): Promise<Record<string, any>> => {
    return apiRequest.get('/data/storage/stats');
  },

  /**
   * 获取可用股票代码列表
   */
  getSymbols: (params: GetSymbolsParams = {}): Promise<SymbolListResponse> => {
    return apiRequest.get('/data/symbols', params);
  },

  /**
   * 获取指定股票的历史数据
   */
  getStockData: (params: GetStockDataParams): Promise<StockDataResponse> => {
    const { symbol, ...otherParams } = params;
    return apiRequest.get(`/data/stocks/${symbol}`, otherParams);
  },

  /**
   * 直接从AKShare获取股票数据
   */
  getAKShareStockData: (
    symbol: string,
    days: number = 30,
    saveToStorage: boolean = false
  ): Promise<StockDataResponse> => {
    return apiRequest.get(`/data/akshare/stocks/${symbol}`, {
      days,
      save_to_storage: saveToStorage,
    });
  },

  /**
   * 创建测试数据 (仅开发模式)
   */
  createSampleData: (
    symbol: string = 'TEST001'
  ): Promise<{
    message: string;
    symbol: string;
    bars_count: number;
    date_range: string;
  }> => {
    return apiRequest.post('/data/test/sample-data', { symbol });
  },
};

export default dataApi;

