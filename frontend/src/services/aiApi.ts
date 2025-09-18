/**
 * AI助手API调用服务
 *
 * 提供与AI助手的对话功能，与后端/api/v1/ai接口对接
 */

import { apiRequest } from './api';
import type {
  ChatMessage,
  ChatRequest,
  ChatResponse,
  PresetCategoryResponse,
  AnalysisRequest,
  AnalysisResponse,
} from '../types/api';

/**
 * AI助手API接口
 */
export const aiApi = {
  /**
   * 与AI助手聊天
   * @param request 聊天请求
   * @returns AI回复
   */
  async chatWithAI(request: ChatRequest): Promise<ChatResponse | null> {
    try {
      const response = await apiRequest.post<ChatResponse>('/ai/chat', request);
      return response;
    } catch (error) {
      console.error('AI聊天失败:', error);

      // 提供fallback模拟回复，防止完全无法使用
      const fallbackMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'assistant',
        content:
          '抱歉，AI服务暂时不可用，请稍后再试。我是您的量化交易助手，可以为您分析市场、优化策略和提供投资建议。',
        timestamp: new Date().toISOString(),
        suggestions: [
          '查看市场数据',
          '分析策略表现',
          '获取投资建议',
          '风险评估',
        ],
      };

      return {
        message: fallbackMessage,
        status: 'fallback',
        processing_time: 0,
      };
    }
  },

  /**
   * 获取预设问题分类
   * @returns 预设问题分类列表
   */
  async getPresetQuestions(): Promise<PresetCategoryResponse | null> {
    try {
      const response =
        await apiRequest.get<PresetCategoryResponse>('/ai/presets');
      return response;
    } catch (error) {
      console.error('获取预设问题失败:', error);

      // 提供fallback模拟数据
      return {
        categories: [
          {
            category: '市场分析',
            icon: 'BarChartOutlined',
            questions: [
              {
                title: '当前市场趋势分析',
                description: '分析当前A股市场的整体趋势和热点板块',
                prompt:
                  '请分析当前A股市场的整体趋势，包括主要指数走势、热点板块和市场情绪指标。',
              },
            ],
          },
          {
            category: '策略优化',
            icon: 'LineChartOutlined',
            questions: [
              {
                title: '策略参数调优',
                description: '优化现有量化策略的参数设置',
                prompt:
                  '我的移动平均线策略最近表现不佳，请帮我分析可能的原因并建议参数优化方案。',
              },
            ],
          },
        ],
        total_count: 2,
      };
    }
  },

  /**
   * 获取聊天历史
   * @param sessionId 会话ID
   * @param limit 消息数量限制
   * @returns 聊天历史
   */
  async getChatHistory(
    sessionId: string,
    limit?: number
  ): Promise<{
    session_id: string;
    messages: ChatMessage[];
    total_count: number;
    message?: string;
  } | null> {
    try {
      const params = { session_id: sessionId, limit: limit || 50 };
      const response = await apiRequest.get('/ai/chat/history', params);
      return response;
    } catch (error) {
      console.error('获取聊天历史失败:', error);

      // 返回空历史记录
      return {
        session_id: sessionId,
        messages: [],
        total_count: 0,
        message: '聊天历史功能暂时不可用',
      };
    }
  },

  /**
   * 智能分析服务
   * @param request 分析请求
   * @returns 分析结果
   */
  async getAnalysis(
    request: AnalysisRequest
  ): Promise<AnalysisResponse | null> {
    try {
      const response = await apiRequest.post<AnalysisResponse>(
        '/ai/analysis',
        request
      );
      return response;
    } catch (error) {
      console.error('智能分析失败:', error);

      // 提供fallback响应
      return {
        analysis_type: request.type,
        results: {
          message: '分析服务暂时不可用，请稍后再试',
          fallback: true,
        },
        recommendations: [
          '请稍后重试分析功能',
          '可以查看历史分析报告',
          '联系技术支持获取帮助',
        ],
      };
    }
  },

  /**
   * 获取AI助手使用统计
   * @returns 使用统计信息
   */
  async getAIStats(): Promise<{
    total_conversations: number;
    total_messages: number;
    popular_categories: Array<{
      category: string;
      count: number;
    }>;
    response_time_avg: number;
    user_satisfaction: number;
    last_updated: string;
  } | null> {
    try {
      const response = await apiRequest.get('/ai/stats/summary');
      return response;
    } catch (error) {
      console.error('获取AI统计失败:', error);

      // 提供fallback数据
      return {
        total_conversations: 0,
        total_messages: 0,
        popular_categories: [
          { category: '市场分析', count: 0 },
          { category: '策略优化', count: 0 },
          { category: '投研报告', count: 0 },
        ],
        response_time_avg: 1.5,
        user_satisfaction: 4.0,
        last_updated: new Date().toISOString(),
      };
    }
  },

  /**
   * 生成会话ID
   * @returns 新的会话ID
   */
  generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },
};

export default aiApi;
