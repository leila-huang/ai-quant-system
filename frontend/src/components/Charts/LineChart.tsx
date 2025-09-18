import { useMemo, forwardRef } from 'react';
import BaseChart from './BaseChart';
import type { LineChartProps, ChartRef } from '@/types/charts';
import {
  convertTimeSeriesData,
  generateLineChartOption,
  formatChartValue,
} from '@/utils/chartUtils';

/**
 * 折线图组件
 * 用于显示时间序列数据，如收益曲线、价格趋势等
 */
const LineChart = forwardRef<ChartRef, LineChartProps>(
  (
    {
      data,
      title,
      xAxisLabel,
      yAxisLabel,
      showGrid = true,
      smooth = true,
      area = false,
      color,
      formatter,
      theme = 'light',
      loading = false,
      width,
      height,
      className,
      onChartReady,
      onClick,
      onDataZoom,
    },
    ref
  ) => {
    // 生成图表配置
    const chartOption = useMemo(() => {
      if (!data || data.length === 0) {
        return {
          title: {
            text: '暂无数据',
            left: 'center',
            top: 'middle',
            textStyle: {
              color: theme === 'dark' ? '#ffffff' : '#666666',
              fontSize: 16,
            },
          },
        };
      }

      const [xData, yData] = convertTimeSeriesData(data);

      const option = generateLineChartOption(xData, yData, {
        title,
        smooth,
        area,
        color,
        theme,
      });

      // 自定义坐标轴标签
      if (xAxisLabel) {
        option.xAxis = {
          ...option.xAxis,
          name: xAxisLabel,
          nameTextStyle: {
            color: theme === 'dark' ? '#ffffff' : '#333333',
          },
        };
      }

      if (yAxisLabel) {
        option.yAxis = {
          ...option.yAxis,
          name: yAxisLabel,
          nameTextStyle: {
            color: theme === 'dark' ? '#ffffff' : '#333333',
          },
        };
      }

      // 自定义网格显示
      if (
        option.grid &&
        typeof option.grid === 'object' &&
        !Array.isArray(option.grid)
      ) {
        (option.grid as any).show = showGrid;
      }

      // 自定义tooltip格式化
      if (
        option.tooltip &&
        typeof option.tooltip === 'object' &&
        !Array.isArray(option.tooltip)
      ) {
        (option.tooltip as any).formatter = (params: any) => {
          if (Array.isArray(params)) {
            params = params[0];
          }

          const value = formatter
            ? formatter(params.value)
            : formatChartValue(params.value, 'number', 2);

          return `
            <div style="padding: 8px;">
              <div style="margin-bottom: 4px; font-weight: bold;">
                ${params.name}
              </div>
              <div style="display: flex; align-items: center;">
                <span style="
                  display: inline-block;
                  width: 10px;
                  height: 10px;
                  background-color: ${params.color};
                  border-radius: 50%;
                  margin-right: 6px;
                "></span>
                <span>${title || '数值'}: ${value}</span>
              </div>
            </div>
          `;
        };
      }

      // 添加数据缩放功能
      option.dataZoom = [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
        {
          show: data.length > 20, // 数据点多时显示缩放条
          type: 'slider',
          start: 0,
          end: 100,
          bottom: 30,
        },
      ];

      return option;
    }, [
      data,
      title,
      xAxisLabel,
      yAxisLabel,
      showGrid,
      smooth,
      area,
      color,
      formatter,
      theme,
    ]);

    return (
      <BaseChart
        ref={ref}
        option={chartOption}
        width={width}
        height={height}
        className={`line-chart ${className || ''}`}
        loading={loading}
        theme={theme}
        onChartReady={onChartReady}
        onClick={onClick}
        onDataZoom={onDataZoom}
      />
    );
  }
);

LineChart.displayName = 'LineChart';

export default LineChart;
