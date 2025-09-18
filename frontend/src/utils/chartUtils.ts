// ECharts图表工具函数

import type { EChartsOption } from 'echarts';
import type {
  ChartTheme,
  TimeSeriesDataPoint,
  CandlestickDataPoint,
  BarDataPoint,
  ResponsiveConfig,
} from '@/types/charts';

// 默认图表主题
export const chartThemes: Record<'light' | 'dark', ChartTheme> = {
  light: {
    backgroundColor: '#ffffff',
    textColor: '#333333',
    gridColor: '#f0f0f0',
    upColor: '#ef4444', // 红色（涨）
    downColor: '#22c55e', // 绿色（跌）
    lineColor: '#3b82f6',
    areaColors: ['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'],
  },
  dark: {
    backgroundColor: '#1f1f1f',
    textColor: '#ffffff',
    gridColor: '#333333',
    upColor: '#ef4444',
    downColor: '#22c55e',
    lineColor: '#60a5fa',
    areaColors: ['#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa'],
  },
};

// 获取响应式尺寸配置
export const getResponsiveConfig = (): ResponsiveConfig => ({
  xs: { width: '100%', height: 200 },
  sm: { width: '100%', height: 250 },
  md: { width: '100%', height: 300 },
  lg: { width: '100%', height: 350 },
  xl: { width: '100%', height: 400 },
  xxl: { width: '100%', height: 450 },
});

// 格式化数值
export const formatChartValue = (
  value: number,
  type: 'number' | 'percent' | 'currency' | 'volume' = 'number',
  decimals: number = 2
): string => {
  if (isNaN(value) || value === null || value === undefined) return '--';

  switch (type) {
    case 'percent':
      return `${(value * 100).toFixed(decimals)}%`;
    case 'currency':
      return `¥${value.toLocaleString('zh-CN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })}`;
    case 'volume':
      if (value >= 1e8) {
        return `${(value / 1e8).toFixed(1)}亿`;
      }
      if (value >= 1e4) {
        return `${(value / 1e4).toFixed(1)}万`;
      }
      return value.toFixed(0);
    default:
      return value.toLocaleString('zh-CN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      });
  }
};

// 转换时间序列数据为ECharts格式
export const convertTimeSeriesData = (
  data: TimeSeriesDataPoint[]
): [string[], number[]] => {
  const xData: string[] = [];
  const yData: number[] = [];

  data.forEach(point => {
    xData.push(
      typeof point.timestamp === 'string'
        ? point.timestamp
        : new Date(point.timestamp).toLocaleDateString('zh-CN')
    );
    yData.push(point.value);
  });

  return [xData, yData];
};

// 转换K线数据为ECharts格式
export const convertCandlestickData = (
  data: CandlestickDataPoint[]
): {
  xData: string[];
  candlestickData: number[][];
  volumeData: number[];
} => {
  const xData: string[] = [];
  const candlestickData: number[][] = [];
  const volumeData: number[] = [];

  data.forEach(point => {
    xData.push(
      typeof point.timestamp === 'string'
        ? point.timestamp
        : new Date(point.timestamp).toLocaleDateString('zh-CN')
    );
    candlestickData.push([point.open, point.close, point.low, point.high]);
    volumeData.push(point.volume || 0);
  });

  return { xData, candlestickData, volumeData };
};

// 转换柱状图数据为ECharts格式
export const convertBarData = (
  data: BarDataPoint[]
): { xData: string[]; yData: number[] } => {
  const xData = data.map(point => point.name);
  const yData = data.map(point => point.value);
  return { xData, yData };
};

// 计算移动平均线
export const calculateMA = (
  data: number[],
  period: number
): (number | null)[] => {
  const result: (number | null)[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
  }

  return result;
};

// 生成基础图表配置
export const generateBaseChartOption = (
  theme: 'light' | 'dark' = 'light'
): EChartsOption => {
  const currentTheme = chartThemes[theme];

  return {
    backgroundColor: currentTheme.backgroundColor,
    textStyle: {
      color: currentTheme.textColor,
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
      borderColor: currentTheme.gridColor,
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: currentTheme.backgroundColor,
      borderColor: currentTheme.gridColor,
      textStyle: {
        color: currentTheme.textColor,
      },
    },
    legend: {
      textStyle: {
        color: currentTheme.textColor,
      },
    },
    xAxis: {
      type: 'category',
      axisLine: {
        lineStyle: {
          color: currentTheme.gridColor,
        },
      },
      axisLabel: {
        color: currentTheme.textColor,
      },
      splitLine: {
        show: false,
      },
    },
    yAxis: {
      type: 'value',
      axisLine: {
        lineStyle: {
          color: currentTheme.gridColor,
        },
      },
      axisLabel: {
        color: currentTheme.textColor,
        formatter: (value: number) => formatChartValue(value),
      },
      splitLine: {
        lineStyle: {
          color: currentTheme.gridColor,
          type: 'dashed',
        },
      },
    },
    toolbox: {
      feature: {
        saveAsImage: { title: '保存图片' },
        restore: { title: '重置' },
        dataZoom: { title: { zoom: '区域缩放', back: '区域缩放还原' } },
      },
      iconStyle: {
        borderColor: currentTheme.textColor,
      },
      emphasis: {
        iconStyle: {
          borderColor: currentTheme.lineColor,
        },
      },
    },
  };
};

// 生成折线图配置
export const generateLineChartOption = (
  xData: string[],
  yData: number[],
  options: {
    title?: string;
    smooth?: boolean;
    area?: boolean;
    color?: string;
    theme?: 'light' | 'dark';
  } = {}
): EChartsOption => {
  const baseOption = generateBaseChartOption(options.theme);
  const theme = chartThemes[options.theme || 'light'];

  return {
    ...baseOption,
    title: options.title
      ? {
          text: options.title,
          textStyle: { color: theme.textColor },
        }
      : undefined,
    xAxis: {
      ...baseOption.xAxis,
      data: xData,
    },
    series: [
      {
        type: 'line',
        data: yData,
        smooth: options.smooth || false,
        itemStyle: {
          color: options.color || theme.lineColor,
        },
        lineStyle: {
          color: options.color || theme.lineColor,
        },
        areaStyle: options.area
          ? {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  { offset: 0, color: options.color || theme.lineColor },
                  {
                    offset: 1,
                    color: 'rgba(255, 255, 255, 0.1)',
                  },
                ],
              },
            }
          : undefined,
      },
    ],
  };
};

// 生成K线图配置
export const generateCandlestickChartOption = (
  xData: string[],
  candlestickData: number[][],
  volumeData: number[],
  options: {
    title?: string;
    showVolume?: boolean;
    showMA?: boolean;
    maPeriods?: number[];
    theme?: 'light' | 'dark';
  } = {}
): EChartsOption => {
  const baseOption = generateBaseChartOption(options.theme);
  const theme = chartThemes[options.theme || 'light'];

  const series: any[] = [
    {
      name: 'K线',
      type: 'candlestick',
      data: candlestickData,
      itemStyle: {
        color: theme.upColor,
        color0: theme.downColor,
        borderColor: theme.upColor,
        borderColor0: theme.downColor,
      },
    },
  ];

  // 添加移动平均线
  if (options.showMA && options.maPeriods) {
    const closeData = candlestickData.map(item => item[1]); // 收盘价
    options.maPeriods.forEach((period, index) => {
      const maData = calculateMA(closeData, period);
      series.push({
        name: `MA${period}`,
        type: 'line',
        data: maData,
        smooth: true,
        lineStyle: {
          width: 1,
          color: theme.areaColors[index % theme.areaColors.length],
        },
        showSymbol: false,
      });
    });
  }

  // 添加成交量
  if (options.showVolume) {
    series.push({
      name: '成交量',
      type: 'bar',
      yAxisIndex: 1,
      data: volumeData,
      itemStyle: {
        color: (params: any) => {
          const candlestick = candlestickData[params.dataIndex];
          return candlestick[1] >= candlestick[0]
            ? theme.upColor
            : theme.downColor;
        },
      },
    });
  }

  return {
    ...baseOption,
    title: options.title
      ? {
          text: options.title,
          textStyle: { color: theme.textColor },
        }
      : undefined,
    xAxis: {
      ...baseOption.xAxis,
      data: xData,
    },
    yAxis: options.showVolume
      ? [
          baseOption.yAxis as any,
          {
            type: 'value',
            scale: true,
            axisLabel: {
              color: theme.textColor,
              formatter: (value: number) => formatChartValue(value, 'volume'),
            },
            splitLine: { show: false },
            axisLine: {
              lineStyle: { color: theme.gridColor },
            },
          } as any,
        ]
      : baseOption.yAxis,
    series,
    dataZoom: [
      {
        type: 'inside',
        start: 80,
        end: 100,
      },
      {
        show: true,
        type: 'slider',
        start: 80,
        end: 100,
      },
    ],
  };
};

// 生成柱状图配置
export const generateBarChartOption = (
  xData: string[],
  yData: number[],
  options: {
    title?: string;
    horizontal?: boolean;
    color?: string | string[];
    showValues?: boolean;
    theme?: 'light' | 'dark';
  } = {}
): EChartsOption => {
  const baseOption = generateBaseChartOption(options.theme);
  const theme = chartThemes[options.theme || 'light'];

  const isHorizontal = options.horizontal || false;

  return {
    ...baseOption,
    title: options.title
      ? {
          text: options.title,
          textStyle: { color: theme.textColor },
        }
      : undefined,
    xAxis: isHorizontal
      ? ({ type: 'value' } as any)
      : { ...baseOption.xAxis, data: xData },
    yAxis: isHorizontal
      ? ({ type: 'category', data: xData } as any)
      : baseOption.yAxis,
    series: [
      {
        type: 'bar',
        data: yData,
        itemStyle: {
          color:
            typeof options.color === 'string' ? options.color : theme.lineColor,
        } as any,
        label: options.showValues
          ? {
              show: true,
              position: isHorizontal ? 'right' : 'top',
              color: theme.textColor,
              formatter: (params: any) => formatChartValue(params.value),
            }
          : undefined,
      } as any,
    ],
  };
};

// 防抖函数用于图表resize
export const debounceResize = (func: () => void, wait: number = 300) => {
  let timeout: NodeJS.Timeout;
  return () => {
    clearTimeout(timeout);
    timeout = setTimeout(func, wait);
  };
};

// 获取当前屏幕断点
export const getCurrentBreakpoint = (): keyof ResponsiveConfig => {
  const width = window.innerWidth;
  if (width < 576) return 'xs';
  if (width < 768) return 'sm';
  if (width < 992) return 'md';
  if (width < 1200) return 'lg';
  if (width < 1600) return 'xl';
  return 'xxl';
};
