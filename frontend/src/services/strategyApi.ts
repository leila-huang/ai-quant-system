/**
 * 策略管理API调用服务
 *
 * 提供策略的CRUD操作和控制功能，与后端/api/v1/strategies接口对接
 */

import { apiRequest } from './api';
import type {
  Strategy,
  StrategyCreateRequest,
  StrategyUpdateRequest,
  StrategyListResponse,
  StrategyActionResponse,
} from '../types/api';

/**
 * 策略管理API接口
 */
export const strategyApi = {
  /**
   * 获取策略列表
   * @param params 查询参数
   * @returns 策略列表响应
   */
  async getStrategies(params?: {
    page?: number;
    page_size?: number;
    status?: string;
    search?: string;
  }): Promise<StrategyListResponse> {
    try {
      const response = await apiRequest.get<StrategyListResponse>(
        '/strategies',
        params
      );
      return response;
    } catch (error) {
      console.error('获取策略列表失败:', error);

      // 提供fallback模拟数据，防止完全无法使用
      return {
        strategies: [],
        total_count: 0,
        page: params?.page || 1,
        page_size: params?.page_size || 20,
      };
    }
  },

  /**
   * 获取单个策略详情
   * @param strategyId 策略ID
   * @returns 策略详情
   */
  async getStrategy(strategyId: string | number): Promise<Strategy | null> {
    try {
      const response = await apiRequest.get<Strategy>(
        `/strategies/${strategyId}`
      );
      return response;
    } catch (error) {
      console.error('获取策略详情失败:', error);
      return null;
    }
  },

  /**
   * 创建新策略
   * @param strategyData 策略创建数据
   * @returns 创建的策略
   */
  async createStrategy(
    strategyData: StrategyCreateRequest
  ): Promise<Strategy | null> {
    try {
      const response = await apiRequest.post<Strategy>(
        '/strategies',
        strategyData
      );
      return response;
    } catch (error) {
      console.error('创建策略失败:', error);
      throw error; // 创建操作失败应该向上抛出异常
    }
  },

  /**
   * 更新策略
   * @param strategyId 策略ID
   * @param updateData 更新数据
   * @returns 更新后的策略
   */
  async updateStrategy(
    strategyId: string | number,
    updateData: StrategyUpdateRequest
  ): Promise<Strategy | null> {
    try {
      const response = await apiRequest.put<Strategy>(
        `/strategies/${strategyId}`,
        updateData
      );
      return response;
    } catch (error) {
      console.error('更新策略失败:', error);
      throw error; // 更新操作失败应该向上抛出异常
    }
  },

  /**
   * 删除策略
   * @param strategyId 策略ID
   * @returns 删除操作结果
   */
  async deleteStrategy(
    strategyId: string | number
  ): Promise<StrategyActionResponse | null> {
    try {
      const response = await apiRequest.delete<StrategyActionResponse>(
        `/strategies/${strategyId}`
      );
      return response;
    } catch (error) {
      console.error('删除策略失败:', error);
      throw error; // 删除操作失败应该向上抛出异常
    }
  },

  /**
   * 启动策略
   * @param strategyId 策略ID
   * @returns 操作结果
   */
  async startStrategy(
    strategyId: string | number
  ): Promise<StrategyActionResponse | null> {
    try {
      const response = await apiRequest.post<StrategyActionResponse>(
        `/strategies/${strategyId}/start`
      );
      return response;
    } catch (error) {
      console.error('启动策略失败:', error);
      throw error; // 控制操作失败应该向上抛出异常
    }
  },

  /**
   * 停止策略
   * @param strategyId 策略ID
   * @returns 操作结果
   */
  async stopStrategy(
    strategyId: string | number
  ): Promise<StrategyActionResponse | null> {
    try {
      const response = await apiRequest.post<StrategyActionResponse>(
        `/strategies/${strategyId}/stop`
      );
      return response;
    } catch (error) {
      console.error('停止策略失败:', error);
      throw error; // 控制操作失败应该向上抛出异常
    }
  },

  /**
   * 暂停策略
   * @param strategyId 策略ID
   * @returns 操作结果
   */
  async pauseStrategy(
    strategyId: string | number
  ): Promise<StrategyActionResponse | null> {
    try {
      const response = await apiRequest.post<StrategyActionResponse>(
        `/strategies/${strategyId}/pause`
      );
      return response;
    } catch (error) {
      console.error('暂停策略失败:', error);
      throw error; // 控制操作失败应该向上抛出异常
    }
  },

  /**
   * 获取策略统计概览
   * @returns 统计信息
   */
  async getStrategyStats(): Promise<{
    total_strategies: number;
    active_strategies: number;
    paused_strategies: number;
    stopped_strategies: number;
    last_updated: string;
  } | null> {
    try {
      const response = await apiRequest.get<{
        total_strategies: number;
        active_strategies: number;
        paused_strategies: number;
        stopped_strategies: number;
        last_updated: string;
      }>('/strategies/stats/summary');
      return response;
    } catch (error) {
      console.error('获取策略统计失败:', error);

      // 提供fallback模拟数据
      return {
        total_strategies: 0,
        active_strategies: 0,
        paused_strategies: 0,
        stopped_strategies: 0,
        last_updated: new Date().toISOString(),
      };
    }
  },
};

export default strategyApi;
