import { useState, useEffect } from 'react';
import { Card, Statistic, Row, Col, Switch, Typography, Alert } from 'antd';
import {
  DashboardOutlined,
  ClockCircleOutlined,
  AreaChartOutlined,
} from '@ant-design/icons';
import { PerformanceMonitor } from '@/utils/performance';

const { Text } = Typography;

interface PerformanceMetrics {
  chartInit: number;
  chartRender: number;
  componentRender: number;
  fcp: number;
  lcp: number;
  memoryUsage: any;
}

/**
 * 性能监控组件
 * 仅在开发环境或启用性能监控时显示
 */
const PerformanceMonitorComponent: React.FC = () => {
  const [visible, setVisible] = useState(import.meta.env.DEV);
  const [metrics, setMetrics] = useState<Partial<PerformanceMetrics>>({});
  const [isEnabled, setIsEnabled] = useState(false);

  useEffect(() => {
    if (!isEnabled) return;

    const monitor = PerformanceMonitor.getInstance();
    const interval = setInterval(() => {
      const chartInitStats = monitor.getMetricStats('chart_init');
      const chartRenderStats = monitor.getMetricStats('chart_render');
      const componentRenderStats = monitor.getMetricStats('component_render');
      const fcpStats = monitor.getMetricStats('FCP');
      const lcpStats = monitor.getMetricStats('LCP');

      // 获取内存使用情况
      const memoryInfo = (performance as any).memory;

      setMetrics({
        chartInit: chartInitStats?.avg || 0,
        chartRender: chartRenderStats?.avg || 0,
        componentRender: componentRenderStats?.avg || 0,
        fcp: fcpStats?.avg || 0,
        lcp: lcpStats?.avg || 0,
        memoryUsage: memoryInfo
          ? {
              used: Math.round(memoryInfo.usedJSHeapSize / 1024 / 1024),
              total: Math.round(memoryInfo.totalJSHeapSize / 1024 / 1024),
              limit: Math.round(memoryInfo.jsHeapSizeLimit / 1024 / 1024),
            }
          : null,
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [isEnabled]);

  if (!visible) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 1000,
        width: '300px',
      }}
    >
      <Card
        size="small"
        title={
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <span>
              <DashboardOutlined /> 性能监控
            </span>
            <Switch
              size="small"
              checked={isEnabled}
              onChange={setIsEnabled}
              checkedChildren="开启"
              unCheckedChildren="关闭"
            />
          </div>
        }
        extra={
          <Text
            type="secondary"
            style={{ cursor: 'pointer' }}
            onClick={() => setVisible(false)}
          >
            ✕
          </Text>
        }
      >
        {!isEnabled ? (
          <Alert
            message="性能监控已关闭"
            description="开启开关以开始监控性能指标"
            type="info"
            showIcon
          />
        ) : (
          <>
            <Row gutter={[8, 8]}>
              <Col span={12}>
                <Statistic
                  title="图表初始化"
                  value={metrics.chartInit}
                  precision={1}
                  suffix="ms"
                  prefix={<ClockCircleOutlined />}
                  valueStyle={{
                    fontSize: '12px',
                    color:
                      (metrics.chartInit || 0) > 100 ? '#cf1322' : '#3f8600',
                  }}
                />
              </Col>

              <Col span={12}>
                <Statistic
                  title="图表渲染"
                  value={metrics.chartRender}
                  precision={1}
                  suffix="ms"
                  prefix={<AreaChartOutlined />}
                  valueStyle={{
                    fontSize: '12px',
                    color:
                      (metrics.chartRender || 0) > 200 ? '#cf1322' : '#3f8600',
                  }}
                />
              </Col>

              <Col span={12}>
                <Statistic
                  title="FCP"
                  value={metrics.fcp}
                  precision={0}
                  suffix="ms"
                  valueStyle={{
                    fontSize: '12px',
                    color: (metrics.fcp || 0) > 2000 ? '#cf1322' : '#3f8600',
                  }}
                />
              </Col>

              <Col span={12}>
                <Statistic
                  title="LCP"
                  value={metrics.lcp}
                  precision={0}
                  suffix="ms"
                  valueStyle={{
                    fontSize: '12px',
                    color: (metrics.lcp || 0) > 3000 ? '#cf1322' : '#3f8600',
                  }}
                />
              </Col>

              {metrics.memoryUsage && (
                <>
                  <Col span={24}>
                    <div
                      style={{
                        borderTop: '1px solid #f0f0f0',
                        paddingTop: '8px',
                        marginTop: '8px',
                      }}
                    >
                      <Text strong style={{ fontSize: '12px' }}>
                        内存使用情况
                      </Text>
                    </div>
                  </Col>

                  <Col span={12}>
                    <Statistic
                      title="已用内存"
                      value={metrics.memoryUsage.used}
                      suffix="MB"
                      valueStyle={{ fontSize: '12px' }}
                    />
                  </Col>

                  <Col span={12}>
                    <Statistic
                      title="内存限制"
                      value={metrics.memoryUsage.limit}
                      suffix="MB"
                      valueStyle={{ fontSize: '12px' }}
                    />
                  </Col>
                </>
              )}
            </Row>

            <div
              style={{
                marginTop: '8px',
                borderTop: '1px solid #f0f0f0',
                paddingTop: '8px',
              }}
            >
              <Text type="secondary" style={{ fontSize: '10px' }}>
                仅开发环境显示 • 数据每2秒更新
              </Text>
            </div>
          </>
        )}
      </Card>
    </div>
  );
};

export default PerformanceMonitorComponent;
