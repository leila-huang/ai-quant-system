import { useEffect, useState } from 'react';
import { Card, Row, Col, Spin, Alert, Button, Space } from 'antd';
import { ReloadOutlined, BarChartOutlined } from '@ant-design/icons';
import { LineChart, BarChart } from './Charts';
import type { DataStatusResponse } from '@/types/api';
import type { TimeSeriesDataPoint, BarDataPoint } from '@/types/charts';
import { dataApi } from '@/services';

interface SystemOverviewProps {
  className?: string;
  theme?: 'light' | 'dark';
}

/**
 * 系统概览数据可视化组件
 * 展示系统运行状态、数据趋势等图表
 */
const SystemOverview: React.FC<SystemOverviewProps> = ({
  className,
  theme = 'light',
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [systemData, setSystemData] = useState<DataStatusResponse | null>(null);

  // 生成模拟的系统性能数据
  const generateSystemTrendData = (): TimeSeriesDataPoint[] => {
    const data: TimeSeriesDataPoint[] = [];
    const now = new Date();

    for (let i = 6; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);

      data.push({
        timestamp: date.toLocaleDateString('zh-CN'),
        value: Math.random() * 100 + 50, // 模拟系统负载 50-150%
      });
    }
    return data;
  };

  // 生成存储分布数据
  const generateStorageDistribution = (): BarDataPoint[] => {
    if (!systemData) return [];

    return systemData.storage_info.map((storage, index) => ({
      name: storage.type.toUpperCase(),
      value: storage.total_size_mb,
      color: ['#1890ff', '#52c41a', '#faad14', '#722ed1'][index % 4],
      symbols_count: storage.symbols_count,
      total_files: storage.total_files,
    }));
  };

  // 获取系统数据
  const fetchSystemData = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await dataApi.getStatus();
      setSystemData(data);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : '获取系统数据失败';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // 组件挂载时获取数据
  useEffect(() => {
    fetchSystemData();
  }, []);

  // 处理刷新
  const handleRefresh = () => {
    fetchSystemData();
  };

  if (error) {
    return (
      <Card className={className}>
        <Alert
          message="数据加载失败"
          description={error}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={handleRefresh}>
              重试
            </Button>
          }
        />
      </Card>
    );
  }

  const systemTrendData = generateSystemTrendData();
  const storageDistributionData = generateStorageDistribution();

  return (
    <div className={className}>
      <Row gutter={[16, 16]}>
        {/* 系统性能趋势 */}
        <Col xs={24} lg={12}>
          <Card
            title="系统负载趋势"
            size="small"
            extra={
              <Space>
                <Button
                  type="text"
                  icon={<ReloadOutlined />}
                  onClick={handleRefresh}
                  loading={loading}
                  size="small"
                />
              </Space>
            }
          >
            {loading ? (
              <div
                style={{
                  height: 250,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Spin size="large" tip="加载中..." />
              </div>
            ) : (
              <LineChart
                data={systemTrendData}
                title="系统负载"
                xAxisLabel="日期"
                yAxisLabel="负载率(%)"
                height={250}
                smooth
                area
                theme={theme}
                color="#1890ff"
                formatter={value => `${value.toFixed(1)}%`}
              />
            )}
          </Card>
        </Col>

        {/* 存储分布 */}
        <Col xs={24} lg={12}>
          <Card
            title="存储分布"
            size="small"
            extra={<BarChartOutlined style={{ color: '#666' }} />}
          >
            {loading ? (
              <div
                style={{
                  height: 250,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Spin size="large" tip="加载中..." />
              </div>
            ) : (
              <BarChart
                data={storageDistributionData}
                title="存储容量"
                xAxisLabel="存储类型"
                yAxisLabel="容量(MB)"
                height={250}
                showValues
                theme={theme}
                color={['#1890ff', '#52c41a', '#faad14', '#722ed1']}
                formatter={value => `${(value / 1024).toFixed(2)} GB`}
              />
            )}
          </Card>
        </Col>

        {/* 数据源健康状态 */}
        <Col xs={24}>
          <Card title="数据源健康状态" size="small">
            {loading ? (
              <Spin tip="加载中...">
                <div style={{ height: 100 }} />
              </Spin>
            ) : systemData ? (
              <div style={{ padding: '16px 0' }}>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                    gap: '16px',
                  }}
                >
                  {systemData.data_sources.map((source, index) => {
                    const healthScore =
                      source.enabled && source.status === 'available' ? 100 : 0;
                    const color = healthScore === 100 ? '#52c41a' : '#ff4d4f';

                    return (
                      <div
                        key={index}
                        style={{
                          padding: '12px',
                          border: '1px solid #f0f0f0',
                          borderRadius: '6px',
                          textAlign: 'center',
                        }}
                      >
                        <div
                          style={{
                            fontSize: '24px',
                            fontWeight: 'bold',
                            color,
                            marginBottom: '4px',
                          }}
                        >
                          {healthScore}%
                        </div>
                        <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                          {source.name}
                        </div>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          {source.enabled ? source.status : '已禁用'}
                        </div>
                        {source.last_update && (
                          <div
                            style={{
                              fontSize: '11px',
                              color: '#999',
                              marginTop: '4px',
                            }}
                          >
                            {new Date(source.last_update).toLocaleString(
                              'zh-CN'
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <Alert message="暂无数据" type="info" />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SystemOverview;
