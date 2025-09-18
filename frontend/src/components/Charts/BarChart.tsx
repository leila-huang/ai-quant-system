import { useMemo, forwardRef } from 'react';
import BaseChart from './BaseChart';
import type { BarChartProps, ChartRef } from '@/types/charts';
import {
  convertBarData,
  generateBarChartOption,
  formatChartValue,
} from '@/utils/chartUtils';

/**
 * 柱状图组件
 * 用于显示分类数据对比，如成交量、持仓分布等
 */
const BarChart = forwardRef<ChartRef, BarChartProps>(
  (
    {
      data,
      title,
      xAxisLabel,
      yAxisLabel,
      horizontal = false,
      showValues = true,
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

      const { xData, yData } = convertBarData(data);

      const option = generateBarChartOption(xData, yData, {
        title,
        horizontal,
        color: Array.isArray(color) ? color : color,
        showValues,
        theme,
      });

      // 自定义坐标轴标签
      if (xAxisLabel) {
        const xAxisConfig = Array.isArray(option.xAxis)
          ? option.xAxis[0]
          : option.xAxis;
        if (xAxisConfig && typeof xAxisConfig === 'object') {
          Object.assign(xAxisConfig, {
            name: xAxisLabel,
            nameTextStyle: {
              color: theme === 'dark' ? '#ffffff' : '#333333',
            },
          });
        }
      }

      if (yAxisLabel) {
        const yAxisConfig = Array.isArray(option.yAxis)
          ? option.yAxis[0]
          : option.yAxis;
        if (yAxisConfig && typeof yAxisConfig === 'object') {
          Object.assign(yAxisConfig, {
            name: yAxisLabel,
            nameTextStyle: {
              color: theme === 'dark' ? '#ffffff' : '#333333',
            },
          });
        }
      }

      // 自定义多色柱状图
      if (
        Array.isArray(color) &&
        option.series &&
        Array.isArray(option.series)
      ) {
        const barSeries = option.series[0];
        if (barSeries && typeof barSeries === 'object') {
          (barSeries as any).itemStyle = {
            color: (params: any) => {
              return color[params.dataIndex % color.length];
            },
          };
        }
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

          const dataPoint = data[params.dataIndex];

          let content = `
            <div style="padding: 8px;">
              <div style="margin-bottom: 4px; font-weight: bold;">
                ${params.name}
              </div>
              <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="
                  display: inline-block;
                  width: 10px;
                  height: 10px;
                  background-color: ${params.color};
                  border-radius: 2px;
                  margin-right: 6px;
                "></span>
                <span>${title || '数值'}: ${value}</span>
              </div>
          `;

          // 显示额外的数据属性
          if (dataPoint && typeof dataPoint === 'object') {
            Object.entries(dataPoint).forEach(([key, val]) => {
              if (key !== 'name' && key !== 'value' && key !== 'color') {
                content += `
                  <div style="margin-left: 16px; color: #666; font-size: 12px;">
                    ${key}: ${val}
                  </div>
                `;
              }
            });
          }

          content += `</div>`;
          return content;
        };
      }

      // 添加动画效果
      option.animation = true;
      option.animationDuration = 1000;
      option.animationEasing = 'cubicOut';

      // 数据较多时添加缩放功能
      if (data.length > 20) {
        option.dataZoom = [
          {
            type: 'inside',
            start: 0,
            end: 100,
          },
          {
            show: true,
            type: 'slider',
            start: 0,
            end: 100,
            [horizontal ? 'yAxisIndex' : 'xAxisIndex']: 0,
          },
        ];
      }

      return option;
    }, [
      data,
      title,
      xAxisLabel,
      yAxisLabel,
      horizontal,
      showValues,
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
        className={`bar-chart ${horizontal ? 'horizontal' : 'vertical'} ${className || ''}`}
        loading={loading}
        theme={theme}
        onChartReady={onChartReady}
        onClick={onClick}
        onDataZoom={onDataZoom}
      />
    );
  }
);

BarChart.displayName = 'BarChart';

export default BarChart;
