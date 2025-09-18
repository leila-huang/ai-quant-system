// API 类型定义 - 基于后端 Pydantic 模型

// ===================== 数据管理API类型 =====================

export interface DataSourceInfo {
  name: string;
  enabled: boolean;
  status: string;
  last_update?: string;
}

export interface StorageInfo {
  type: string;
  path: string;
  total_files: number;
  total_size_mb: number;
  symbols_count: number;
}

export interface DataStatusResponse {
  data_sources: DataSourceInfo[];
  storage_info: StorageInfo[];
  last_sync?: string;
}

export interface SymbolListResponse {
  symbols: string[];
  total_count: number;
  data_source: string;
}

export interface BasicStockBar {
  date: string; // ISO date string
  open_price: number;
  close_price: number;
  high_price: number;
  low_price: number;
  volume: number;
  amount?: number;
}

export interface StockDataResponse {
  symbol: string;
  name?: string;
  bars: BasicStockBar[];
  total_count: number;
  start_date?: string;
  end_date?: string;
}

// ===================== 回测API类型 =====================

export interface StrategyConfig {
  strategy_type: 'ma_crossover' | 'rsi_mean_reversion' | 'momentum' | 'custom';
  strategy_name: string;
  parameters: Record<string, any>;
}

export interface BacktestRequest {
  backtest_name: string;
  strategy_config: StrategyConfig;
  universe: string[];
  start_date: string; // ISO date string
  end_date: string; // ISO date string

  // 交易配置
  initial_capital: number;
  commission: number;
  enable_cost_model: boolean;
  broker_type: 'standard' | 'low_commission' | 'minimum';

  // A股约束配置
  enable_constraints: boolean;
  price_limit_enabled: boolean;
  t_plus_1_enabled: boolean;

  // 回测配置
  rebalance_frequency: 'daily' | 'weekly' | 'monthly';
  benchmark_symbol?: string;
  async_execution: boolean;
}

export interface BacktestResponse {
  backtest_id: string;
  backtest_name: string;
  status: string;
  performance_metrics: Record<string, number>;
  execution_time_seconds: number;
  trade_count: number;
  total_return: number;
  annual_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  task_id?: string;
}

export interface BacktestStatusResponse {
  backtest_id: string;
  status: string;
  progress: number;
  current_step: string;
  estimated_time_remaining?: number;
  error_message?: string;
  started_at: string;
  updated_at: string;
}

// 回测元数据
export interface BacktestMetadata {
  backtest_id: string;
  backtest_name: string;
  strategy_config: StrategyConfig;
  universe: string[];
  start_date: string;
  end_date: string;
  status: string;
  performance_metrics: Record<string, number>;
  started_at: string;
  completed_at?: string;
  execution_time_seconds?: number;
  trade_count?: number;
  total_return?: number;
  annual_return?: number;
  max_drawdown?: number;
  sharpe_ratio?: number;
}

export interface BacktestListResponse {
  backtests: BacktestMetadata[];
  total_count: number;
}

export interface ReportRequest {
  backtest_id: string;
  report_format: 'html' | 'pdf' | 'json' | 'excel';
  include_charts: boolean;
  include_trades: boolean;
  custom_title?: string;
}

// 回测结果详情
export interface BacktestResultDetail {
  backtest_id: string;
  metadata: BacktestMetadata;
  results: {
    returns: Record<string, number>;
    trades: any[];
    positions: any[];
    metrics: Record<string, number>;
    metadata: Record<string, any>;
  };
  status: string;
}

// 策略参数信息
export interface StrategyParameter {
  type: string;
  default: any;
  description: string;
  min?: number;
  max?: number;
  options?: any[];
}

// 策略信息
export interface StrategyInfo {
  name: string;
  description: string;
  parameters: Record<string, StrategyParameter>;
}

// 支持的策略响应
export interface SupportedStrategiesResponse {
  supported_strategies: Record<string, StrategyInfo>;
  total_count: number;
}

// 回测对比请求
export interface BacktestComparisonRequest {
  backtest_ids: string[];
  comparison_metrics?: string[];
  report_format?: string;
}

// ===================== 通用API类型 =====================

export interface ApiResponse<T = any> {
  data?: T;
  message?: string;
  error?: string;
  status_code?: number;
}

export interface ApiError {
  detail: string;
  status_code: number;
}

// ===================== 请求参数类型 =====================

export interface GetStockDataParams {
  symbol: string;
  start_date?: string;
  end_date?: string;
  limit?: number;
}

export interface GetSymbolsParams {
  source?: 'akshare' | 'storage';
  limit?: number;
  [key: string]: unknown;
}

export interface BacktestListParams {
  status_filter?: string;
  start_date?: string;
  limit?: number;
  [key: string]: unknown;
}

// ===================== 策略管理API类型 =====================

export interface Strategy {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'paused' | 'stopped' | 'backtest';
  description?: string;

  // 绩效指标
  totalReturn: number;
  annualReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;

  // 时间字段
  createdAt: string; // ISO格式字符串
  lastRunAt?: string; // ISO格式字符串

  // 配置参数
  parameters: Record<string, any>;

  // 风控参数
  max_position_size: number;
  max_daily_loss: number;
  max_drawdown_limit: number;

  // 版本和代码信息
  version: string;
  code?: string;
}

export interface StrategyCreateRequest {
  name: string;
  description?: string;
  type: string;
  config: Record<string, any>;
  code?: string;
  version?: string;
  max_position_size?: number;
  max_daily_loss?: number;
  max_drawdown?: number;
}

export interface StrategyUpdateRequest {
  name?: string;
  description?: string;
  type?: string;
  config?: Record<string, any>;
  code?: string;
  version?: string;
  max_position_size?: number;
  max_daily_loss?: number;
  max_drawdown?: number;
}

export interface StrategyListResponse {
  strategies: Strategy[];
  total_count: number;
  page: number;
  page_size: number;
}

export interface StrategyActionResponse {
  success: boolean;
  message: string;
  strategy_id: string;
  new_status?: string;
}

// ===================== AI助手API类型 =====================

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string; // ISO格式字符串
  suggestions?: string[];
}

export interface ChatRequest {
  message: string;
  context?: Record<string, any>;
  session_id?: string;
}

export interface ChatResponse {
  message: ChatMessage;
  status: string;
  processing_time: number;
}

export interface PresetQuestionItem {
  title: string;
  description: string;
  prompt: string;
}

export interface PresetCategory {
  category: string;
  icon: string;
  questions: PresetQuestionItem[];
}

export interface PresetCategoryResponse {
  categories: PresetCategory[];
  total_count: number;
}

export interface AnalysisRequest {
  type: 'market' | 'strategy' | 'stock' | 'factor';
  params: Record<string, any>;
}

export interface AnalysisResponse {
  analysis_type: string;
  results: Record<string, any>;
  charts?: Array<Record<string, any>>;
  recommendations: string[];
}

// ===================== 健康检查类型 =====================

export interface HealthResponse {
  status: string;
  timestamp: string;
  version?: string;
  database?: {
    status: string;
    connection_count?: number;
  };
  redis?: {
    status: string;
    memory_usage?: string;
  };
}
