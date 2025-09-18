import {
  useRef,
  useEffect,
  useImperativeHandle,
  forwardRef,
  memo,
} from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import { Spin } from 'antd';
import type { BaseChartProps, ChartRef } from '@/types/charts';
import {
  debounceResize,
  getCurrentBreakpoint,
  getResponsiveConfig,
} from '@/utils/chartUtils';
import { PerformanceMonitor } from '@/utils/performance';

/**
 * ECharts基础图表组件
 * 提供统一的图表封装，包括响应式、加载状态、主题等功能
 */
const BaseChart = forwardRef<
  ChartRef,
  BaseChartProps & { option: EChartsOption }
>(
  (
    {
      option,
      width = '100%',
      height = 400,
      className = '',
      loading = false,
      theme = 'light',
      onChartReady,
      onClick,
      onDataZoom,
    },
    ref
  ) => {
    const chartRef = useRef<ReactECharts>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const performanceMonitor = PerformanceMonitor.getInstance();

    // 暴露图表实例方法
    useImperativeHandle(ref, () => ({
      getChart: () => chartRef.current?.getEchartsInstance(),
      resize: () => {
        const chart = chartRef.current?.getEchartsInstance();
        if (chart) {
          chart.resize();
        }
      },
      dispatchAction: (payload: any) => {
        const chart = chartRef.current?.getEchartsInstance();
        if (chart) {
          chart.dispatchAction(payload);
        }
      },
      setOption: (newOption: EChartsOption, notMerge?: boolean) => {
        const chart = chartRef.current?.getEchartsInstance();
        if (chart) {
          chart.setOption(newOption, notMerge);
        }
      },
    }));

    // 处理响应式尺寸
    const getResponsiveSize = () => {
      const breakpoint = getCurrentBreakpoint();
      const responsiveConfig = getResponsiveConfig();
      return responsiveConfig[breakpoint];
    };

    // 监听窗口大小变化
    useEffect(() => {
      const handleResize = debounceResize(() => {
        const chart = chartRef.current?.getEchartsInstance();
        if (chart) {
          chart.resize();
        }
      });

      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }, []);

    // 图表就绪回调
    const handleChartReady = (chart: any) => {
      // 性能监控 - 记录图表初始化时间
      const initStart = performance.now();

      if (onChartReady) {
        onChartReady(chart);
      }

      // 绑定点击事件
      if (onClick) {
        chart.on('click', onClick);
      }

      // 绑定缩放事件
      if (onDataZoom) {
        chart.on('datazoom', onDataZoom);
      }

      // 记录初始化完成时间
      const initEnd = performance.now();
      performanceMonitor.addMetric('chart_init', initEnd - initStart);

      // 监控图表渲染性能
      chart.on('finished', () => {
        const renderEnd = performance.now();
        performanceMonitor.addMetric('chart_render', renderEnd - initStart);
      });
    };

    // 计算实际尺寸
    const actualSize =
      typeof width === 'string' && width === '100%'
        ? getResponsiveSize()
        : { width, height };

    return (
      <div
        ref={containerRef}
        className={`chart-container ${className}`}
        style={{
          width: actualSize.width,
          height: actualSize.height,
          position: 'relative',
        }}
      >
        {loading && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(255, 255, 255, 0.8)',
              zIndex: 10,
            }}
          >
            <Spin size="large" tip="图表加载中..." />
          </div>
        )}

        <ReactECharts
          ref={chartRef}
          option={option}
          style={{
            width: '100%',
            height: '100%',
          }}
          theme={theme}
          onChartReady={handleChartReady}
          opts={{
            renderer: 'canvas', // 使用canvas渲染，性能更好
            // useDirtyRect: true, // 开启脏矩形优化 - 暂时注释掉直到类型支持
          }}
        />
      </div>
    );
  }
);

// 使用memo优化性能，避免不必要的重渲染
const MemoizedBaseChart = memo(BaseChart, (prevProps, nextProps) => {
  // 自定义比较函数，深度比较option对象
  if (JSON.stringify(prevProps.option) !== JSON.stringify(nextProps.option)) {
    return false;
  }

  // 比较其他props
  return (
    prevProps.width === nextProps.width &&
    prevProps.height === nextProps.height &&
    prevProps.loading === nextProps.loading &&
    prevProps.theme === nextProps.theme &&
    prevProps.className === nextProps.className
  );
});

MemoizedBaseChart.displayName = 'BaseChart';

export default MemoizedBaseChart;
