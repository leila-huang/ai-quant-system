// ECharts图表组件类型定义

import type { EChartsOption } from 'echarts';

// 基础图表配置
export interface BaseChartProps {
  width?: string | number;
  height?: string | number;
  className?: string;
  loading?: boolean;
  theme?: 'light' | 'dark';
  onChartReady?: (chart: any) => void;
  onClick?: (params: any) => void;
  onDataZoom?: (params: any) => void;
}

// 时间序列数据点
export interface TimeSeriesDataPoint {
  timestamp: string | number;
  value: number;
  [key: string]: any;
}

// K线数据点 [timestamp, open, close, low, high, volume]
export interface CandlestickDataPoint {
  timestamp: string | number;
  open: number;
  close: number;
  low: number;
  high: number;
  volume?: number;
  [key: string]: any;
}

// 柱状图数据点
export interface BarDataPoint {
  name: string;
  value: number;
  color?: string;
  [key: string]: any;
}

// 折线图属性
export interface LineChartProps extends BaseChartProps {
  data: TimeSeriesDataPoint[];
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  showGrid?: boolean;
  smooth?: boolean;
  area?: boolean;
  color?: string;
  formatter?: (value: number) => string;
}

// K线图属性
export interface CandlestickChartProps extends BaseChartProps {
  data: CandlestickDataPoint[];
  title?: string;
  showVolume?: boolean;
  showMA?: boolean;
  maPeriods?: number[];
  upColor?: string;
  downColor?: string;
  formatter?: {
    price?: (value: number) => string;
    volume?: (value: number) => string;
    tooltip?: (params: any) => string;
  };
}

// 柱状图属性
export interface BarChartProps extends BaseChartProps {
  data: BarDataPoint[];
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  horizontal?: boolean;
  showValues?: boolean;
  color?: string | string[];
  formatter?: (value: number) => string;
}

// 图表主题配置
export interface ChartTheme {
  backgroundColor: string;
  textColor: string;
  gridColor: string;
  upColor: string;
  downColor: string;
  lineColor: string;
  areaColors: string[];
}

// 图表工具栏配置
export interface ChartToolbox {
  show: boolean;
  feature?: {
    saveAsImage?: boolean;
    restore?: boolean;
    dataView?: boolean;
    dataZoom?: boolean;
    magicType?: string[];
  };
}

// 图表数据缩放配置
export interface ChartDataZoom {
  show: boolean;
  type?: 'slider' | 'inside';
  start?: number;
  end?: number;
  filterMode?: 'filter' | 'weakFilter' | 'empty' | 'none';
}

// 图表图例配置
export interface ChartLegend {
  show: boolean;
  type?: 'plain' | 'scroll';
  orient?: 'horizontal' | 'vertical';
  left?: string | number;
  top?: string | number;
  right?: string | number;
  bottom?: string | number;
}

// 完整的图表选项配置
export interface ChartOptions extends Omit<EChartsOption, 'series'> {
  theme?: ChartTheme;
  toolbox?: ChartToolbox;
  dataZoom?: ChartDataZoom | ChartDataZoom[];
  legend?: ChartLegend;
}

// 图表实例引用
export interface ChartRef {
  getChart: () => any;
  resize: () => void;
  dispatchAction: (payload: any) => void;
  setOption: (option: EChartsOption, notMerge?: boolean) => void;
}

// 响应式配置
export interface ResponsiveConfig {
  xs: { width: string | number; height: string | number };
  sm: { width: string | number; height: string | number };
  md: { width: string | number; height: string | number };
  lg: { width: string | number; height: string | number };
  xl: { width: string | number; height: string | number };
  xxl: { width: string | number; height: string | number };
}

// 图表数据更新配置
export interface ChartUpdateConfig {
  animation?: boolean;
  animationDuration?: number;
  animationEasing?: string;
  notMerge?: boolean;
  replaceMerge?: string[];
}

// 性能优化配置
export interface ChartPerformanceConfig {
  progressive?: number;
  progressiveThreshold?: number;
  useUTC?: boolean;
  hoverLayerThreshold?: number;
}

