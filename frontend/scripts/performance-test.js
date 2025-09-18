#!/usr/bin/env node

/**
 * 性能测试脚本
 * 测试页面加载时间、图表渲染性能等关键指标
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const TEST_CONFIG = {
  url: process.env.TEST_URL || 'http://localhost:3000',
  timeout: 60000,
  iterations: 3,
  thresholds: {
    pageLoadTime: 3000, // 3秒
    chartRenderTime: 1000, // 1秒
    fcp: 2000, // First Contentful Paint - 2秒
    lcp: 3000, // Largest Contentful Paint - 3秒
    cls: 0.1, // Cumulative Layout Shift
    fid: 100, // First Input Delay - 100ms
  },
};

class PerformanceTester {
  constructor() {
    this.browser = null;
    this.results = [];
  }

  async initialize() {
    this.browser = await puppeteer.launch({
      headless: 'new',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--disable-extensions',
        '--disable-plugins',
        '--disable-images',
        '--disable-javascript-harmony-shipping',
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
      ],
    });
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
    }
  }

  async measurePageLoad(page, url, pageName) {
    console.log(`📊 测试页面: ${pageName}`);

    const startTime = Date.now();

    // 开始性能监控
    await page.evaluateOnNewDocument(() => {
      window.performanceMetrics = {
        navigationStart: Date.now(),
        loadComplete: null,
        firstPaint: null,
        firstContentfulPaint: null,
        largestContentfulPaint: null,
      };

      // 监听性能事件
      new PerformanceObserver(list => {
        for (const entry of list.getEntries()) {
          if (entry.name === 'first-paint') {
            window.performanceMetrics.firstPaint = entry.startTime;
          }
          if (entry.name === 'first-contentful-paint') {
            window.performanceMetrics.firstContentfulPaint = entry.startTime;
          }
        }
      }).observe({ entryTypes: ['paint'] });

      new PerformanceObserver(list => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];
        window.performanceMetrics.largestContentfulPaint = lastEntry.startTime;
      }).observe({ entryTypes: ['largest-contentful-paint'] });
    });

    // 导航到页面
    const response = await page.goto(url, {
      waitUntil: 'networkidle2',
      timeout: TEST_CONFIG.timeout,
    });

    if (!response.ok()) {
      throw new Error(`Failed to load ${url}: ${response.status()}`);
    }

    const loadTime = Date.now() - startTime;

    // 等待页面完全渲染
    await page.waitForTimeout(2000);

    // 获取性能指标
    const metrics = await page.evaluate(() => {
      const paintMetrics = performance.getEntriesByType('paint');
      const navigation = performance.getEntriesByType('navigation')[0];

      return {
        loadTime: Date.now() - window.performanceMetrics.navigationStart,
        domContentLoaded:
          navigation.domContentLoadedEventEnd - navigation.navigationStart,
        firstPaint: window.performanceMetrics.firstPaint || 0,
        firstContentfulPaint:
          window.performanceMetrics.firstContentfulPaint || 0,
        largestContentfulPaint:
          window.performanceMetrics.largestContentfulPaint || 0,
        domElements: document.querySelectorAll('*').length,
        memoryUsage: performance.memory
          ? {
              usedJSHeapSize: performance.memory.usedJSHeapSize,
              totalJSHeapSize: performance.memory.totalJSHeapSize,
              jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
            }
          : null,
      };
    });

    // 获取浏览器指标
    const browserMetrics = await page.metrics();

    const result = {
      pageName,
      url,
      timestamp: new Date().toISOString(),
      loadTime,
      ...metrics,
      browserMetrics: {
        JSEventListeners: browserMetrics.JSEventListeners,
        Nodes: browserMetrics.Nodes,
        JSHeapUsedSize: browserMetrics.JSHeapUsedSize,
        JSHeapTotalSize: browserMetrics.JSHeapTotalSize,
      },
    };

    console.log(`  ⏱️  页面加载时间: ${loadTime}ms`);
    console.log(
      `  🎨 首次内容绘制: ${metrics.firstContentfulPaint.toFixed(0)}ms`
    );
    console.log(
      `  🏗️  最大内容绘制: ${metrics.largestContentfulPaint.toFixed(0)}ms`
    );
    console.log(`  📦 DOM元素数量: ${metrics.domElements}`);

    return result;
  }

  async measureChartPerformance(page) {
    console.log(`📈 测试图表渲染性能`);

    // 注入性能测试代码
    await page.evaluate(() => {
      window.chartPerformanceMetrics = [];

      // 监听图表渲染事件
      if (window.echarts) {
        const originalInit = window.echarts.init;
        window.echarts.init = function (...args) {
          const startTime = performance.now();
          const chart = originalInit.apply(this, args);

          chart.on('finished', () => {
            const renderTime = performance.now() - startTime;
            window.chartPerformanceMetrics.push({
              type: 'chart_render',
              renderTime,
              timestamp: Date.now(),
            });
          });

          return chart;
        };
      }
    });

    // 等待图表渲染
    await page.waitForTimeout(5000);

    // 获取图表性能数据
    const chartMetrics = await page.evaluate(() => {
      return window.chartPerformanceMetrics || [];
    });

    if (chartMetrics.length > 0) {
      const avgRenderTime =
        chartMetrics.reduce((sum, m) => sum + m.renderTime, 0) /
        chartMetrics.length;
      console.log(`  📊 平均图表渲染时间: ${avgRenderTime.toFixed(2)}ms`);
      return {
        averageChartRenderTime: avgRenderTime,
        chartCount: chartMetrics.length,
      };
    }

    return { averageChartRenderTime: 0, chartCount: 0 };
  }

  async measureInteractionPerformance(page) {
    console.log(`🖱️  测试交互响应时间`);

    const interactions = [];

    try {
      // 测试按钮点击响应
      const buttons = await page.$$('button');
      if (buttons.length > 0) {
        const startTime = Date.now();
        await buttons[0].click();
        await page.waitForTimeout(100);
        const responseTime = Date.now() - startTime;
        interactions.push({ type: 'button_click', responseTime });
        console.log(`  🔘 按钮点击响应: ${responseTime}ms`);
      }

      // 测试输入框响应
      const inputs = await page.$$('input[type="text"]');
      if (inputs.length > 0) {
        const startTime = Date.now();
        await inputs[0].focus();
        await inputs[0].type('测试输入');
        const responseTime = Date.now() - startTime;
        interactions.push({ type: 'input_response', responseTime });
        console.log(`  ⌨️  输入响应时间: ${responseTime}ms`);
      }
    } catch (error) {
      console.warn(`  ⚠️  交互测试部分失败: ${error.message}`);
    }

    return interactions;
  }

  async runTest(iteration) {
    console.log(`\n🚀 开始第 ${iteration + 1} 次测试`);

    const page = await this.browser.newPage();

    // 设置viewport
    await page.setViewport({ width: 1920, height: 1080 });

    // 设置用户代理
    await page.setUserAgent(
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    );

    const results = [];

    try {
      // 测试各个页面
      const pages = [
        { path: '/', name: 'Dashboard' },
        { path: '/strategy', name: 'Strategy' },
        { path: '/backtest', name: 'Backtest' },
        { path: '/trading', name: 'Trading' },
        { path: '/ai', name: 'AI' },
      ];

      for (const pageConfig of pages) {
        const url = `${TEST_CONFIG.url}${pageConfig.path}`;

        try {
          const pageResult = await this.measurePageLoad(
            page,
            url,
            pageConfig.name
          );

          // 测试图表性能（仅对包含图表的页面）
          if (
            ['Dashboard', 'Strategy', 'Backtest', 'Trading'].includes(
              pageConfig.name
            )
          ) {
            const chartResult = await this.measureChartPerformance(page);
            pageResult.chartMetrics = chartResult;
          }

          // 测试交互性能
          const interactionResult =
            await this.measureInteractionPerformance(page);
          pageResult.interactions = interactionResult;

          results.push(pageResult);

          // 页面间等待
          await page.waitForTimeout(1000);
        } catch (error) {
          console.error(`❌ ${pageConfig.name} 页面测试失败:`, error.message);
          results.push({
            pageName: pageConfig.name,
            url,
            error: error.message,
            timestamp: new Date().toISOString(),
          });
        }
      }
    } finally {
      await page.close();
    }

    return results;
  }

  async runAllTests() {
    console.log('🏁 开始性能测试');
    console.log(`测试URL: ${TEST_CONFIG.url}`);
    console.log(`测试轮次: ${TEST_CONFIG.iterations}`);

    for (let i = 0; i < TEST_CONFIG.iterations; i++) {
      try {
        const iterationResults = await this.runTest(i);
        this.results.push(...iterationResults);
      } catch (error) {
        console.error(`❌ 第 ${i + 1} 次测试失败:`, error.message);
      }
    }
  }

  generateReport() {
    console.log('\n📋 生成测试报告');

    // 按页面分组结果
    const pageGroups = {};
    this.results.forEach(result => {
      if (!result.error) {
        if (!pageGroups[result.pageName]) {
          pageGroups[result.pageName] = [];
        }
        pageGroups[result.pageName].push(result);
      }
    });

    const report = {
      testConfig: TEST_CONFIG,
      timestamp: new Date().toISOString(),
      summary: {},
      details: pageGroups,
      passed: true,
      failures: [],
    };

    // 计算各页面的平均指标
    Object.keys(pageGroups).forEach(pageName => {
      const results = pageGroups[pageName];
      const metrics = {
        averageLoadTime:
          results.reduce((sum, r) => sum + r.loadTime, 0) / results.length,
        averageFCP:
          results.reduce((sum, r) => sum + (r.firstContentfulPaint || 0), 0) /
          results.length,
        averageLCP:
          results.reduce((sum, r) => sum + (r.largestContentfulPaint || 0), 0) /
          results.length,
        averageChartRenderTime: 0,
        testCount: results.length,
      };

      // 图表渲染时间
      const chartResults = results.filter(r => r.chartMetrics);
      if (chartResults.length > 0) {
        metrics.averageChartRenderTime =
          chartResults.reduce(
            (sum, r) => sum + r.chartMetrics.averageChartRenderTime,
            0
          ) / chartResults.length;
      }

      report.summary[pageName] = metrics;

      // 检查性能阈值
      const failures = [];
      if (metrics.averageLoadTime > TEST_CONFIG.thresholds.pageLoadTime) {
        failures.push(
          `${pageName}: 页面加载时间超出阈值 (${metrics.averageLoadTime.toFixed(0)}ms > ${TEST_CONFIG.thresholds.pageLoadTime}ms)`
        );
        report.passed = false;
      }
      if (metrics.averageFCP > TEST_CONFIG.thresholds.fcp) {
        failures.push(
          `${pageName}: FCP超出阈值 (${metrics.averageFCP.toFixed(0)}ms > ${TEST_CONFIG.thresholds.fcp}ms)`
        );
        report.passed = false;
      }
      if (metrics.averageLCP > TEST_CONFIG.thresholds.lcp) {
        failures.push(
          `${pageName}: LCP超出阈值 (${metrics.averageLCP.toFixed(0)}ms > ${TEST_CONFIG.thresholds.lcp}ms)`
        );
        report.passed = false;
      }
      if (
        metrics.averageChartRenderTime > TEST_CONFIG.thresholds.chartRenderTime
      ) {
        failures.push(
          `${pageName}: 图表渲染时间超出阈值 (${metrics.averageChartRenderTime.toFixed(0)}ms > ${TEST_CONFIG.thresholds.chartRenderTime}ms)`
        );
        report.passed = false;
      }

      report.failures.push(...failures);
    });

    return report;
  }

  printReport(report) {
    console.log('\n📊 性能测试报告');
    console.log('='.repeat(50));

    console.log(`\n测试时间: ${report.timestamp}`);
    console.log(`总体结果: ${report.passed ? '✅ 通过' : '❌ 失败'}`);

    if (!report.passed) {
      console.log('\n❌ 失败项目:');
      report.failures.forEach(failure => console.log(`  - ${failure}`));
    }

    console.log('\n📈 性能指标摘要:');
    Object.keys(report.summary).forEach(pageName => {
      const metrics = report.summary[pageName];
      console.log(`\n${pageName} 页面:`);
      console.log(
        `  📄 平均加载时间: ${metrics.averageLoadTime.toFixed(0)}ms (阈值: ${TEST_CONFIG.thresholds.pageLoadTime}ms)`
      );
      console.log(
        `  🎨 平均FCP: ${metrics.averageFCP.toFixed(0)}ms (阈值: ${TEST_CONFIG.thresholds.fcp}ms)`
      );
      console.log(
        `  🏗️  平均LCP: ${metrics.averageLCP.toFixed(0)}ms (阈值: ${TEST_CONFIG.thresholds.lcp}ms)`
      );
      if (metrics.averageChartRenderTime > 0) {
        console.log(
          `  📊 平均图表渲染: ${metrics.averageChartRenderTime.toFixed(0)}ms (阈值: ${TEST_CONFIG.thresholds.chartRenderTime}ms)`
        );
      }
      console.log(`  🔄 测试次数: ${metrics.testCount}`);
    });

    console.log('\n💾 详细报告已保存至: performance-report.json');
  }

  async saveReport(report) {
    const reportPath = path.join(process.cwd(), 'performance-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  }
}

async function main() {
  const tester = new PerformanceTester();

  try {
    await tester.initialize();
    await tester.runAllTests();

    const report = tester.generateReport();
    tester.printReport(report);
    await tester.saveReport(report);

    // 退出码基于测试结果
    process.exit(report.passed ? 0 : 1);
  } catch (error) {
    console.error('❌ 测试过程中发生错误:', error);
    process.exit(1);
  } finally {
    await tester.cleanup();
  }
}

// 处理未捕获的异常
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

process.on('uncaughtException', error => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

if (require.main === module) {
  main();
}

module.exports = { PerformanceTester, TEST_CONFIG };

