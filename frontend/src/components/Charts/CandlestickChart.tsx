import { useMemo, forwardRef } from 'react';
import BaseChart from './BaseChart';
import type { CandlestickChartProps, ChartRef } from '@/types/charts';
import {
  convertCandlestickData,
  generateCandlestickChartOption,
  formatChartValue,
} from '@/utils/chartUtils';

/**
 * K线图组件
 * 专业的股票K线图表，支持成交量、移动平均线等功能
 */
const CandlestickChart = forwardRef<ChartRef, CandlestickChartProps>(
  (
    {
      data,
      title,
      showVolume = true,
      showMA = true,
      maPeriods = [5, 10, 20],
      upColor = '#ef4444',
      downColor = '#22c55e',
      formatter,
      theme = 'light',
      loading = false,
      width,
      height = 500, // K线图通常需要更高的高度
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
            text: '暂无K线数据',
            left: 'center',
            top: 'middle',
            textStyle: {
              color: theme === 'dark' ? '#ffffff' : '#666666',
              fontSize: 16,
            },
          },
        };
      }

      const { xData, candlestickData, volumeData } =
        convertCandlestickData(data);

      const option = generateCandlestickChartOption(
        xData,
        candlestickData,
        volumeData,
        {
          title,
          showVolume,
          showMA,
          maPeriods,
          theme,
        }
      );

      // 自定义颜色
      if (option.series && Array.isArray(option.series)) {
        const candlestickSeries = option.series.find(
          (s: any) => s.type === 'candlestick'
        );
        if (candlestickSeries && (candlestickSeries as any).itemStyle) {
          (candlestickSeries as any).itemStyle = {
            ...(candlestickSeries as any).itemStyle,
            color: upColor,
            color0: downColor,
            borderColor: upColor,
            borderColor0: downColor,
          };
        }

        // 更新成交量颜色
        const volumeSeries = option.series.find(
          (s: any) => s.name === '成交量'
        );
        if (
          volumeSeries &&
          (volumeSeries as any).itemStyle &&
          typeof (volumeSeries as any).itemStyle === 'object'
        ) {
          (volumeSeries as any).itemStyle.color = (params: any) => {
            const candlestick = candlestickData[params.dataIndex];
            return candlestick[1] >= candlestick[0] ? upColor : downColor;
          };
        }
      }

      // 自定义tooltip
      if (
        option.tooltip &&
        typeof option.tooltip === 'object' &&
        !Array.isArray(option.tooltip)
      ) {
        (option.tooltip as any).formatter = (params: any) => {
          if (!Array.isArray(params)) {
            params = [params];
          }

          let content = `<div style="padding: 8px; min-width: 200px;">`;
          content += `<div style="margin-bottom: 8px; font-weight: bold; border-bottom: 1px solid #eee; padding-bottom: 4px;">${params[0].name}</div>`;

          params.forEach((param: any) => {
            if (param.seriesType === 'candlestick') {
              const [open, close, low, high] = param.value;
              const change = close - open;
              const changePercent = ((change / open) * 100).toFixed(2);
              const changeColor = change >= 0 ? upColor : downColor;

              content += `
                <div style="margin-bottom: 6px;">
                  <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>开盘:</span>
                    <span>${formatter?.price ? formatter.price(open) : formatChartValue(open, 'currency')}</span>
                  </div>
                  <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>收盘:</span>
                    <span style="color: ${changeColor};">
                      ${formatter?.price ? formatter.price(close) : formatChartValue(close, 'currency')}
                    </span>
                  </div>
                  <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>最高:</span>
                    <span>${formatter?.price ? formatter.price(high) : formatChartValue(high, 'currency')}</span>
                  </div>
                  <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>最低:</span>
                    <span>${formatter?.price ? formatter.price(low) : formatChartValue(low, 'currency')}</span>
                  </div>
                  <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>涨跌:</span>
                    <span style="color: ${changeColor};">
                      ${change >= 0 ? '+' : ''}${formatter?.price ? formatter.price(change) : formatChartValue(change, 'currency')} (${changePercent}%)
                    </span>
                  </div>
                </div>
              `;
            } else if (param.seriesName === '成交量') {
              content += `
                <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                  <span>${param.seriesName}:</span>
                  <span>${formatter?.volume ? formatter.volume(param.value) : formatChartValue(param.value, 'volume')}</span>
                </div>
              `;
            } else if (param.seriesName.startsWith('MA')) {
              content += `
                <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                  <span style="color: ${param.color};">${param.seriesName}:</span>
                  <span>${formatter?.price ? formatter.price(param.value) : formatChartValue(param.value, 'currency')}</span>
                </div>
              `;
            }
          });

          content += `</div>`;

          // 自定义tooltip格式化
          if (formatter?.tooltip) {
            return formatter.tooltip(params);
          }

          return content;
        };
      }

      // 调整网格布局以适应成交量子图
      if (showVolume && option.grid && typeof option.grid === 'object') {
        option.grid = [
          {
            left: '3%',
            right: '4%',
            top: '10%',
            height: '60%',
            containLabel: true,
          },
          {
            left: '3%',
            right: '4%',
            top: '75%',
            height: '20%',
            containLabel: true,
          },
        ];
      }

      return option;
    }, [
      data,
      title,
      showVolume,
      showMA,
      maPeriods,
      upColor,
      downColor,
      formatter,
      theme,
    ]);

    return (
      <BaseChart
        ref={ref}
        option={chartOption}
        width={width}
        height={height}
        className={`candlestick-chart ${className || ''}`}
        loading={loading}
        theme={theme}
        onChartReady={onChartReady}
        onClick={onClick}
        onDataZoom={onDataZoom}
      />
    );
  }
);

CandlestickChart.displayName = 'CandlestickChart';

export default CandlestickChart;
