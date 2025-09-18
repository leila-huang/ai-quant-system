import React from 'react';

/**
 * 性能优化工具函数
 */

// 防抖函数
export const debounce = <T extends (...args: any[]) => void>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

// 节流函数
export const throttle = <T extends (...args: any[]) => void>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

// 检测用户设备性能
export const getDevicePerformance = () => {
  const memory = (navigator as any).deviceMemory || 4; // GB
  const cores = navigator.hardwareConcurrency || 4;

  // 根据设备性能返回配置
  let level: 'low' | 'medium' | 'high';
  if (memory >= 8 && cores >= 8) {
    level = 'high';
  } else if (memory >= 4 && cores >= 4) {
    level = 'medium';
  } else {
    level = 'low';
  }

  return {
    level,
    memory,
    cores,
  };
};

// 性能监控
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: Array<{
    name: string;
    value: number;
    timestamp: number;
  }> = [];

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  // 测量函数执行时间
  measureFunction<T>(name: string, fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const end = performance.now();

    this.addMetric(name, end - start);
    return result;
  }

  // 测量异步函数执行时间
  async measureAsyncFunction<T>(
    name: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const start = performance.now();
    const result = await fn();
    const end = performance.now();

    this.addMetric(name, end - start);
    return result;
  }

  // 添加性能指标
  addMetric(name: string, value: number): void {
    this.metrics.push({
      name,
      value,
      timestamp: Date.now(),
    });

    // 保持最近1000条记录
    if (this.metrics.length > 1000) {
      this.metrics.shift();
    }
  }

  // 获取指标统计
  getMetricStats(name: string) {
    const values = this.metrics.filter(m => m.name === name).map(m => m.value);

    if (values.length === 0) return null;

    const sorted = values.sort((a, b) => a - b);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      p50: sorted[Math.floor(sorted.length * 0.5)],
      p90: sorted[Math.floor(sorted.length * 0.9)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      count: values.length,
    };
  }

  // 获取所有指标
  getAllMetrics() {
    return this.metrics.slice();
  }

  // 清空指标
  clear() {
    this.metrics = [];
  }

  // 清空指标（测试用别名）
  clearMetrics() {
    this.clear();
  }

  // 按名称获取指标
  getMetrics() {
    const grouped: Record<string, number[]> = {};
    this.metrics.forEach(metric => {
      if (!grouped[metric.name]) {
        grouped[metric.name] = [];
      }
      grouped[metric.name].push(metric.value);
    });
    return grouped;
  }

  // 获取平均时间
  getAverageTime(name: string): number {
    const values = this.metrics.filter(m => m.name === name).map(m => m.value);
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  // 获取设备性能信息
  getDevicePerformance() {
    return getDevicePerformance();
  }
}

// Web Vitals 监控
export const measureWebVitals = () => {
  // First Contentful Paint
  if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver(list => {
      list.getEntries().forEach(entry => {
        if (entry.name === 'first-contentful-paint') {
          console.log('FCP:', entry.startTime);
          PerformanceMonitor.getInstance().addMetric('FCP', entry.startTime);
        }
      });
    });

    observer.observe({ entryTypes: ['paint'] });
  }

  // Largest Contentful Paint
  if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver(list => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      console.log('LCP:', lastEntry.startTime);
      PerformanceMonitor.getInstance().addMetric('LCP', lastEntry.startTime);
    });

    observer.observe({ entryTypes: ['largest-contentful-paint'] });
  }
};

// 内存使用监控
export const monitorMemoryUsage = () => {
  if ('memory' in performance) {
    const memory = (performance as any).memory;
    return {
      usedJSHeapSize: memory.usedJSHeapSize,
      totalJSHeapSize: memory.totalJSHeapSize,
      jsHeapSizeLimit: memory.jsHeapSizeLimit,
    };
  }
  return null;
};

// 组件渲染时间监控
export const withPerformanceTracking = <P extends object>(
  WrappedComponent: React.ComponentType<P>,
  componentName: string
) => {
  return React.memo((props: P) => {
    const renderStart = performance.now();

    React.useEffect(() => {
      const renderEnd = performance.now();
      PerformanceMonitor.getInstance().addMetric(
        `${componentName}_render`,
        renderEnd - renderStart
      );
    });

    return React.createElement(WrappedComponent, props);
  });
};

export default {
  debounce,
  throttle,
  getDevicePerformance,
  PerformanceMonitor,
  measureWebVitals,
  monitorMemoryUsage,
  withPerformanceTracking,
};
