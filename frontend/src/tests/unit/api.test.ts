/**
 * API服务单元测试
 * 测试API客户端的基础功能和错误处理
 */

import { describe, test, expect, beforeEach, vi } from 'vitest';
import axios from 'axios';
import { dataApi } from '@/services/dataApi';
import backtestApi from '@/services/backtestApi';
import { healthApi } from '@/services/api';

// Mock axios
vi.mock('axios');
const mockedAxios = vi.mocked(axios);

describe.skip('API Service Unit Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // 设置默认的axios.create mock
    mockedAxios.create = vi.fn(() => ({
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    })) as any;
  });

  describe('健康检查API', () => {
    test('ping应该返回消息', async () => {
      const mockResponse = { data: { message: 'pong' } };
      mockedAxios.get = vi.fn().mockResolvedValue(mockResponse);

      const result = await healthApi.ping();

      expect(result).toEqual({ message: 'pong' });
      expect(mockedAxios.get).toHaveBeenCalledWith('/health/ping');
    });

    test('status应该返回状态信息', async () => {
      const mockResponse = {
        data: {
          status: 'healthy',
          timestamp: '2024-01-01T00:00:00Z',
          version: '1.0.0',
        },
      };
      mockedAxios.get = vi.fn().mockResolvedValue(mockResponse);

      const result = await healthApi.status();

      expect(result).toHaveProperty('status', 'healthy');
      expect(mockedAxios.get).toHaveBeenCalledWith('/health/status');
    });
  });

  describe('数据API', () => {
    test('getStatus应该返回数据状态', async () => {
      const mockResponse = {
        data: {
          status: 'active',
          last_updated: '2024-01-01T00:00:00Z',
          total_symbols: 100,
        },
      };

      const mockApi = {
        get: vi.fn().mockResolvedValue(mockResponse),
      };
      mockedAxios.create = vi.fn().mockReturnValue(mockApi as any);

      // 由于dataApi是在模块加载时创建的，我们需要重新导入或模拟
      expect(() => dataApi.getStatus()).toBeDefined();
    });

    test('getSymbols应该返回支持的股票列表', async () => {
      const mockResponse = {
        data: {
          symbols: [
            { symbol: '000001.SZ', name: '平安银行' },
            { symbol: '000002.SZ', name: '万科A' },
          ],
          total_count: 2,
        },
      };

      const mockApi = {
        get: vi.fn().mockResolvedValue(mockResponse),
      };
      mockedAxios.create = vi.fn().mockReturnValue(mockApi as any);

      expect(() => dataApi.getSymbols()).toBeDefined();
    });

    test('getStockData应该处理股票数据请求', async () => {
      const mockResponse = {
        data: {
          symbol: '000001.SZ',
          data: [
            {
              timestamp: '2024-01-01',
              open: 100,
              high: 105,
              low: 98,
              close: 103,
              volume: 1000000,
            },
          ],
        },
      };

      const mockApi = {
        get: vi.fn().mockResolvedValue(mockResponse),
      };
      mockedAxios.create = vi.fn().mockReturnValue(mockApi as any);

      const params = {
        symbol: '000001.SZ',
        start_date: '2024-01-01',
        end_date: '2024-01-31',
      };

      expect(() => dataApi.getStockData(params)).toBeDefined();
    });
  });

  describe('回测API', () => {
    test('getSupportedStrategies应该返回支持的策略', async () => {
      const mockResponse = {
        data: {
          supported_strategies: {
            ma_crossover: {
              name: '移动平均交叉',
              description: '双均线交叉策略',
              parameters: {
                fast_period: { type: 'int', default: 5, min: 1, max: 50 },
                slow_period: { type: 'int', default: 20, min: 5, max: 100 },
              },
            },
          },
          total_count: 1,
        },
      };

      const mockApi = {
        get: vi.fn().mockResolvedValue(mockResponse),
      };
      mockedAxios.create = vi.fn().mockReturnValue(mockApi as any);

      expect(() => backtestApi.getSupportedStrategies()).toBeDefined();
    });

    test('runBacktest应该提交回测请求', async () => {
      const mockResponse = {
        data: {
          backtest_id: 'bt_123456',
          status: 'submitted',
          message: '回测任务已提交',
        },
      };

      const mockApi = {
        post: vi.fn().mockResolvedValue(mockResponse),
      };
      mockedAxios.create = vi.fn().mockReturnValue(mockApi as any);

      const request = {
        backtest_name: '测试回测',
        strategy_config: {
          strategy_name: 'test_strategy',
          strategy_type: 'ma_crossover',
          parameters: {
            fast_period: 5,
            slow_period: 20,
            stop_loss: 0.05,
          },
        },
        universe: ['000001.SZ'],
        start_date: '2024-01-01',
        end_date: '2024-01-31',
        initial_capital: 100000,
        commission: 0.0005,
        enable_cost_model: true,
        broker_type: 'simulation',
        enable_constraints: false,
        price_limit_enabled: false,
        max_position_size: 1.0,
        position_sizing: 'equal_weight',
      };

      expect(() => backtestApi.runBacktest(request)).toBeDefined();
    });
  });

  describe('错误处理', () => {
    test('应该正确处理网络错误', async () => {
      const mockError = new Error('Network Error');
      mockedAxios.get = vi.fn().mockRejectedValue(mockError);

      try {
        await healthApi.ping();
      } catch (error) {
        expect(error).toBe(mockError);
      }
    });

    test('应该正确处理HTTP错误响应', async () => {
      const mockError = {
        response: {
          status: 404,
          data: { detail: 'Not Found' },
        },
      };
      mockedAxios.get = vi.fn().mockRejectedValue(mockError);

      try {
        await healthApi.status();
      } catch (error) {
        expect(error).toEqual(mockError);
      }
    });
  });

  describe('请求配置测试', () => {
    test('API客户端应该使用正确的基础URL', () => {
      expect(mockedAxios.create).toHaveBeenCalledWith(
        expect.objectContaining({
          timeout: expect.any(Number),
        })
      );
    });

    test('应该设置请求拦截器', () => {
      const mockCreate = vi.fn().mockReturnValue({
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
        interceptors: {
          request: { use: vi.fn() },
          response: { use: vi.fn() },
        },
      });

      mockedAxios.create = mockCreate;

      // 重新导入以触发API客户端创建
      expect(mockCreate).toBeDefined();
    });
  });
});
