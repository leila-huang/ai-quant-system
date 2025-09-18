import { useState } from 'react';
import { Card, Button, Space, Typography, Alert, Divider } from 'antd';
import { healthApi, dataApi, backtestApi } from '@/services';
import { useAppStore, useDataStore } from '@/stores';

const { Title, Paragraph } = Typography;

/**
 * API测试组件 - 用于验证接口调用功能
 */
const ApiTest: React.FC = () => {
  const [testResults, setTestResults] = useState<
    Record<string, { success: boolean; data?: unknown; error?: string }>
  >({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  const { addNotification, setSystemStatus } = useAppStore();
  const { setSymbolList } = useDataStore();

  const runTest = async (testName: string, testFn: () => Promise<unknown>) => {
    setLoading(prev => ({ ...prev, [testName]: true }));

    try {
      const result = await testFn();
      setTestResults(prev => ({
        ...prev,
        [testName]: { success: true, data: result },
      }));

      addNotification({
        type: 'success',
        title: '测试成功',
        message: `${testName} 测试通过`,
      });
    } catch (error: unknown) {
      const errorMessage =
        error && typeof error === 'object' && 'detail' in error
          ? (error as { detail: string }).detail
          : error && typeof error === 'object' && 'message' in error
            ? (error as { message: string }).message
            : 'Unknown error';
      setTestResults(prev => ({
        ...prev,
        [testName]: { success: false, error: errorMessage },
      }));

      addNotification({
        type: 'error',
        title: '测试失败',
        message: `${testName} 测试失败: ${errorMessage}`,
      });
    } finally {
      setLoading(prev => ({ ...prev, [testName]: false }));
    }
  };

  const testHealthPing = () => runTest('健康检查', () => healthApi.ping());

  const testDataStatus = () =>
    runTest('数据状态', async () => {
      const result = await dataApi.getStatus();
      setSystemStatus(result);
      return result;
    });

  const testSymbolList = () =>
    runTest('股票列表', async () => {
      const result = await dataApi.getSymbols({ limit: 10 });
      setSymbolList(result);
      return result;
    });

  const testStockData = () =>
    runTest('股票数据', () =>
      dataApi.getStockData({ symbol: '000001', limit: 10 })
    );

  const testSupportedStrategies = () =>
    runTest('支持策略', () => backtestApi.getSupportedStrategies());

  const renderTestResult = (testName: string) => {
    const result = testResults[testName];
    const isLoading = loading[testName];

    if (isLoading) {
      return <Alert message="测试中..." type="info" showIcon />;
    }

    if (!result) return null;

    if (result.success) {
      return (
        <Alert
          message="测试成功"
          description={
            <details>
              <summary>查看结果</summary>
              <pre
                style={{
                  fontSize: '12px',
                  maxHeight: '200px',
                  overflow: 'auto',
                }}
              >
                {JSON.stringify(result.data, null, 2)}
              </pre>
            </details>
          }
          type="success"
          showIcon
        />
      );
    } else {
      return (
        <Alert
          message="测试失败"
          description={result.error}
          type="error"
          showIcon
        />
      );
    }
  };

  return (
    <Card title="API接口测试" style={{ margin: '16px 0' }}>
      <Paragraph type="secondary">
        此组件用于测试前端与后端API的连接状态。请确保后端服务已启动。
      </Paragraph>

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 健康检查测试 */}
        <div>
          <Title level={4}>1. 健康检查测试</Title>
          <Space>
            <Button
              type="primary"
              onClick={testHealthPing}
              loading={loading.healthPing}
            >
              测试 /api/v1/health/ping
            </Button>
          </Space>
          {renderTestResult('健康检查')}
        </div>

        <Divider />

        {/* 数据管理API测试 */}
        <div>
          <Title level={4}>2. 数据管理API测试</Title>
          <Space wrap>
            <Button onClick={testDataStatus} loading={loading['数据状态']}>
              数据状态
            </Button>
            <Button onClick={testSymbolList} loading={loading['股票列表']}>
              股票列表
            </Button>
            <Button onClick={testStockData} loading={loading['股票数据']}>
              获取股票数据 (000001)
            </Button>
          </Space>
          {renderTestResult('数据状态')}
          {renderTestResult('股票列表')}
          {renderTestResult('股票数据')}
        </div>

        <Divider />

        {/* 回测API测试 */}
        <div>
          <Title level={4}>3. 回测API测试</Title>
          <Space wrap>
            <Button
              onClick={testSupportedStrategies}
              loading={loading['支持策略']}
            >
              支持的策略类型
            </Button>
          </Space>
          {renderTestResult('支持策略')}
        </div>

        <Divider />

        <Alert
          message="测试说明"
          description={
            <ul>
              <li>如果所有测试都通过，说明前端与后端连接正常</li>
              <li>如果测试失败，请检查后端服务是否启动</li>
              <li>测试结果会自动更新到对应的状态管理store中</li>
            </ul>
          }
          type="info"
        />
      </Space>
    </Card>
  );
};

export default ApiTest;
