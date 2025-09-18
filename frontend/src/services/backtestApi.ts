// 回测相关API接口

import { apiRequest } from './api';
import type {
  BacktestRequest,
  BacktestResponse,
  BacktestStatusResponse,
  BacktestListResponse,
  BacktestListParams,
} from '@/types/api';

/**
 * 运行回测
 */
export const runBacktest = (
  request: BacktestRequest
): Promise<BacktestResponse> => {
  return apiRequest.post('/backtest/run', request);
};

/**
 * 查询回测状态
 */
export const getBacktestStatus = (
  backtest_id: string
): Promise<BacktestStatusResponse> => {
  return apiRequest.get(`/backtest/${backtest_id}/status`);
};

/**
 * 获取回测列表
 */
export const getBacktestList = (
  params: BacktestListParams = {}
): Promise<BacktestListResponse> => {
  return apiRequest.get('/backtest/list', params);
};

/**
 * 获取回测详细结果
 */
export const getBacktestResults = (backtest_id: string): Promise<any> => {
  return apiRequest.get(`/backtest/${backtest_id}/results`);
};

/**
 * 生成回测报告
 */
export const generateBacktestReport = (
  backtest_id: string,
  options: {
    report_format?: string;
    include_charts?: boolean;
    include_trades?: boolean;
    custom_title?: string;
  } = {}
): Promise<any> => {
  return apiRequest.post(`/backtest/${backtest_id}/report`, options);
};

/**
 * 对比多个回测结果
 */
export const compareBacktests = (request: {
  backtest_ids: string[];
  comparison_metrics?: string[];
  report_format?: string;
}): Promise<any> => {
  return apiRequest.post('/backtest/compare', request);
};

/**
 * 删除回测
 */
export const deleteBacktest = (backtest_id: string): Promise<any> => {
  return apiRequest.delete(`/backtest/${backtest_id}`);
};

/**
 * 获取支持的策略类型
 */
export const getSupportedStrategies = (): Promise<{
  supported_strategies: Record<string, any>;
  total_count: number;
}> => {
  return apiRequest.get('/backtest/strategies/supported');
};

/**
 * 创建样本回测 (用于测试)
 */
export const createSampleBacktest = (
  name: string = 'SAMPLE_BACKTEST'
): Promise<BacktestResponse> => {
  const sampleRequest: BacktestRequest = {
    backtest_name: name,
    strategy_config: {
      strategy_type: 'ma_crossover',
      strategy_name: '移动平均线穿越策略',
      parameters: {
        fast_period: 5,
        slow_period: 20,
        stop_loss: 0.05,
      },
    },
    universe: ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH', '000858.SZ'],
    start_date: '2023-01-01',
    end_date: '2023-12-31',
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
  };

  return runBacktest(sampleRequest);
};

// 导出所有函数
const backtestApi = {
  runBacktest,
  getBacktestStatus,
  getBacktestList,
  getBacktestResults,
  generateBacktestReport,
  compareBacktests,
  deleteBacktest,
  getSupportedStrategies,
  createSampleBacktest,
};

export default backtestApi;
