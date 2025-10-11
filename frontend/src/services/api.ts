import axios, {
  type AxiosInstance,
  type AxiosResponse,
  type AxiosError,
} from 'axios';
import type { ApiError } from '@/types/api';
import { apiCache, requestManager } from '@/utils/apiCache';

// API 基础配置
const API_BASE_URL = '/api/v1';
const TIMEOUT = 10000; // 10秒超时

// 创建 axios 实例
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
apiClient.interceptors.request.use(
  config => {
    // 添加时间戳防止缓存
    if (config.method === 'get') {
      config.params = {
        ...config.params,
        _t: Date.now(),
      };
    }

    // TODO: 添加认证token (P3阶段实现)
    // const token = getAuthToken();
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }

    return config;
  },
  error => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// 响应拦截器
apiClient.interceptors.response.use(
  (response: AxiosResponse): AxiosResponse => {
    return response;
  },
  (error: AxiosError): Promise<ApiError> => {
    console.error('Response error:', error);

    const apiError: ApiError = {
      detail: '网络请求失败',
      status_code: 500,
    };

    if (error.response) {
      // 服务器响应了错误状态码
      apiError.status_code = error.response.status;
      apiError.detail =
        (error.response.data as { detail?: string })?.detail ||
        error.response.statusText ||
        '服务器错误';
    } else if (error.request) {
      // 请求已发出但没有收到响应
      apiError.detail = '网络连接失败，请检查网络连接';
      apiError.status_code = 0;
    } else {
      // 请求配置错误
      apiError.detail = error.message || '请求配置错误';
    }

    return Promise.reject(apiError);
  }
);

// 通用API请求方法 (带缓存和性能优化)
export const apiRequest = {
  get: <T = unknown>(
    url: string,
    params?: Record<string, unknown>,
    options?: { cache?: boolean; cacheTTL?: number }
  ): Promise<T> => {
    const { cache = true, cacheTTL } = options || {};

    // 检查缓存
    if (cache) {
      const cached = apiCache.get<T>(url, params);
      if (cached !== null) {
        return Promise.resolve(cached);
      }
    }

    // 使用请求管理器执行请求
    return requestManager.queue(async () => {
      const response = await apiClient.get(url, { params });
      const data = response.data as T;

      // 缓存GET请求结果
      if (cache) {
        apiCache.set(url, data, params, cacheTTL);
      }

      return data;
    });
  },

  post: <T = unknown>(url: string, data?: unknown): Promise<T> =>
    requestManager.queue(async () => {
      const response = await apiClient.post(url, data);

      // POST操作后清理相关缓存
      apiCache.delete(url);

      return response.data as T;
    }),

  put: <T = unknown>(url: string, data?: unknown): Promise<T> =>
    requestManager.queue(async () => {
      const response = await apiClient.put(url, data);

      // PUT操作后清理相关缓存
      apiCache.delete(url);

      return response.data as T;
    }),

  delete: <T = unknown>(url: string): Promise<T> =>
    requestManager.queue(async () => {
      const response = await apiClient.delete(url);

      // DELETE操作后清理相关缓存
      apiCache.delete(url);

      return response.data as T;
    }),
};

// 健康检查API
export const healthApi = {
  ping: (): Promise<{ message: string }> => apiRequest.get('/health/ping'),

  status: (): Promise<Record<string, unknown>> =>
    apiRequest.get('/health/status'),
};

// 数据中心API
export const dataAPI = {
  // 获取交易日历
  getTradingCalendar: (year: number): Promise<any> =>
    apiRequest.get('/data/trading-calendar', { year }),

  // 筛选股票
  filterStocks: (filters: any): Promise<any> =>
    apiRequest.post('/data/filter-stocks', filters),

  // 获取财务数据
  getFundamentalData: (symbol: string): Promise<any> =>
    apiRequest.get(`/data/fundamental/${symbol}`),

  // 获取实时行情
  getRealtimeQuote: (symbol: string): Promise<any> =>
    apiRequest.get(`/data/quote/${symbol}`),

  // 获取行业分类
  getIndustryClassification: (symbol: string): Promise<any> =>
    apiRequest.get(`/data/industry/${symbol}`),
};

// 纸上交易API
export const paperTradingAPI = {
  // 创建订单
  createOrder: (order: any): Promise<any> =>
    apiRequest.post('/paper/orders', order),

  // 查询订单
  getOrders: (params?: any): Promise<any> =>
    apiRequest.get('/paper/orders', params),

  // 查询持仓
  getPositions: (symbol?: string): Promise<any> =>
    apiRequest.get('/paper/positions', symbol ? { symbol } : undefined),

  // 组合概览
  getPortfolioSummary: (): Promise<any> =>
    apiRequest.get('/paper/portfolio/summary'),

  // 偏差分析
  getDeviationAnalysis: (backtest_id: string, params?: any): Promise<any> =>
    apiRequest.get('/paper/analysis/deviation', { backtest_id, ...params }),
};

// 组合管理API
export const portfolioAPI = {
  // 信号转权重
  transformSignals: (request: any): Promise<any> =>
    apiRequest.post('/portfolio/signals/transform', request),

  // 组合优化
  optimizePortfolio: (request: any): Promise<any> =>
    apiRequest.post('/portfolio/optimize', request),

  // 权重验证
  validateWeights: (request: any): Promise<any> =>
    apiRequest.post('/portfolio/validate', request),

  // 再平衡
  executeRebalance: (request: any): Promise<any> =>
    apiRequest.post('/portfolio/rebalance', request),

  // 约束模板
  getConstraintTemplates: (): Promise<any> =>
    apiRequest.get('/portfolio/constraints/templates'),
};

export default apiClient;
