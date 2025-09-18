/**
 * 基础性能测试
 * 测试关键功能的性能表现
 */

import { describe, test, expect, beforeEach, vi } from 'vitest';
import { PerformanceMonitor } from '@/utils/performance';

describe('Performance Tests', () => {
  let performanceMonitor: PerformanceMonitor;

  beforeEach(() => {
    performanceMonitor = PerformanceMonitor.getInstance();
    performanceMonitor.clearMetrics();
  });

  describe('性能监控器', () => {
    test('应该能记录性能指标', () => {
      const startTime = performance.now();

      // 模拟一些操作
      for (let i = 0; i < 1000; i++) {
        Math.random();
      }

      const endTime = performance.now();
      const duration = endTime - startTime;

      performanceMonitor.addMetric('test_operation', duration);

      const metrics = performanceMonitor.getMetrics();
      expect(metrics).toHaveProperty('test_operation');
      expect(metrics.test_operation.length).toBe(1);
      expect(metrics.test_operation[0]).toBeCloseTo(duration, 1);
    });

    test('应该能计算平均性能', () => {
      // 添加多个性能指标
      performanceMonitor.addMetric('test_operation', 100);
      performanceMonitor.addMetric('test_operation', 200);
      performanceMonitor.addMetric('test_operation', 300);

      const avg = performanceMonitor.getAverageTime('test_operation');
      expect(avg).toBe(200);
    });

    test('应该能清理性能指标', () => {
      performanceMonitor.addMetric('test_operation', 100);
      expect(performanceMonitor.getMetrics()).toHaveProperty('test_operation');

      performanceMonitor.clearMetrics();
      expect(Object.keys(performanceMonitor.getMetrics())).toHaveLength(0);
    });
  });

  describe('数据处理性能', () => {
    test('大量数据排序性能应该在可接受范围内', () => {
      const startTime = performance.now();

      // 生成大量随机数据
      const data = Array.from({ length: 10000 }, () => Math.random());

      // 排序操作
      data.sort((a, b) => a - b);

      const endTime = performance.now();
      const duration = endTime - startTime;

      // 排序10000个数字应该在100ms以内完成
      expect(duration).toBeLessThan(100);
    });

    test('数组过滤性能应该在可接受范围内', () => {
      const startTime = performance.now();

      // 生成大量数据
      const data = Array.from({ length: 100000 }, (_, i) => ({
        id: i,
        value: Math.random() * 100,
        active: Math.random() > 0.5,
      }));

      // 过滤操作
      const filtered = data.filter(item => item.active && item.value > 50);

      const endTime = performance.now();
      const duration = endTime - startTime;

      // 过滤10万条记录应该在50ms以内完成
      expect(duration).toBeLessThan(50);
      expect(filtered.length).toBeGreaterThan(0);
    });

    test('对象深拷贝性能应该在可接受范围内', () => {
      const startTime = performance.now();

      // 创建复杂对象
      const complexObject = {
        level1: {
          level2: {
            level3: {
              data: Array.from({ length: 1000 }, (_, i) => ({
                id: i,
                name: `item_${i}`,
                values: [i, i * 2, i * 3],
              })),
            },
          },
        },
        metadata: {
          created: Date.now(),
          version: '1.0.0',
        },
      };

      // 深拷贝操作
      const copied = JSON.parse(JSON.stringify(complexObject));

      const endTime = performance.now();
      const duration = endTime - startTime;

      // 深拷贝复杂对象应该在20ms以内完成
      expect(duration).toBeLessThan(20);
      expect(copied).toEqual(complexObject);
      expect(copied).not.toBe(complexObject); // 确保是深拷贝
    });
  });

  describe('计算性能', () => {
    test('数学计算性能应该满足要求', () => {
      const startTime = performance.now();

      // 执行大量数学计算（模拟金融指标计算）
      let result = 0;
      for (let i = 0; i < 100000; i++) {
        const price = 100 + Math.sin(i / 1000) * 10;
        const sma = price * 0.1 + result * 0.9; // 简化的移动平均
        const rsi = Math.abs(Math.sin(i / 100)) * 100; // 简化的RSI
        result += sma + rsi;
      }

      const endTime = performance.now();
      const duration = endTime - startTime;

      // 10万次金融计算应该在100ms以内完成
      expect(duration).toBeLessThan(100);
      expect(result).toBeGreaterThan(0);
    });

    test('日期处理性能应该满足要求', () => {
      const startTime = performance.now();

      // 大量日期操作
      const dates = [];
      const baseDate = new Date('2024-01-01');

      for (let i = 0; i < 10000; i++) {
        const date = new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000);
        dates.push({
          timestamp: date.toISOString(),
          dayOfWeek: date.getDay(),
          isWeekend: date.getDay() === 0 || date.getDay() === 6,
          formatted: date.toLocaleDateString(),
        });
      }

      const endTime = performance.now();
      const duration = endTime - startTime;

      // 1万个日期对象处理应该在50ms以内完成
      expect(duration).toBeLessThan(50);
      expect(dates).toHaveLength(10000);
    });
  });

  describe('内存性能', () => {
    test('应该能检测设备性能级别', () => {
      const devicePerformance = performanceMonitor.getDevicePerformance();

      expect(devicePerformance).toHaveProperty('level');
      expect(['low', 'medium', 'high']).toContain(devicePerformance.level);
      expect(devicePerformance).toHaveProperty('cores');
      expect(devicePerformance).toHaveProperty('memory');
    });

    test('大对象创建和销毁性能', () => {
      const startTime = performance.now();

      // 创建大量对象
      const objects = [];
      for (let i = 0; i < 1000; i++) {
        objects.push({
          id: i,
          data: new Array(100)
            .fill(0)
            .map((_, j) => ({ index: j, value: Math.random() })),
          metadata: {
            created: Date.now(),
            processed: false,
          },
        });
      }

      // 处理对象
      objects.forEach((obj, index) => {
        obj.metadata.processed = true;
        obj.data.sort((a, b) => a.value - b.value);
      });

      const endTime = performance.now();
      const duration = endTime - startTime;

      // 创建和处理1000个复杂对象应该在100ms以内完成
      expect(duration).toBeLessThan(100);
      expect(objects).toHaveLength(1000);
      expect(objects[0].metadata.processed).toBe(true);
    });
  });

  describe('异步操作性能', () => {
    test('Promise处理性能应该满足要求', async () => {
      const startTime = performance.now();

      // 创建大量Promise
      const promises = Array.from(
        { length: 1000 },
        (_, i) =>
          new Promise(resolve => {
            // 模拟异步操作
            setTimeout(() => resolve(i * 2), Math.random() * 10);
          })
      );

      const results = await Promise.all(promises);

      const endTime = performance.now();
      const duration = endTime - startTime;

      expect(results).toHaveLength(1000);
      expect(duration).toBeGreaterThan(0); // 应该花费一些时间
      expect(results[0]).toBe(0);
      expect(results[999]).toBe(1998);
    });
  });
});
