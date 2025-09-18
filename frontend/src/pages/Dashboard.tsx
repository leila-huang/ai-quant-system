import { useEffect, useState, useCallback } from 'react';
import {
  Typography,
  Row,
  Col,
  Card,
  Space,
  Switch,
  Button,
  Alert,
  Divider,
  Tooltip,
} from 'antd';
import {
  DatabaseOutlined,
  CloudServerOutlined,
  LineChartOutlined,
  ReloadOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';

// 导入组件
import ApiTest from '@/components/ApiTest';
import ChartDemo from '@/components/ChartDemo';
import SystemOverview from '@/components/SystemOverview';
import QuickActions from '@/components/QuickActions';
import {
  DataSourceStatusCard,
  StorageStatusCard,
  SystemStatsCard,
} from '@/components/StatusCard';

// 导入状态管理和API
import { useSystemStatus, useNotifications, useAppStore } from '@/stores';
import { dataApi } from '@/services';

const { Title } = Typography;

/**
 * AI量化系统数据概览Dashboard主页面
 * 提供系统状态监控、数据统计、快速操作等功能
 */
const Dashboard: React.FC = () => {
  // 状态管理
  const systemStatus = useSystemStatus();
  const notifications = useNotifications();
  const { setSystemStatus } = useAppStore();

  // 本地状态
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(
    null
  );
  const [lastRefreshTime, setLastRefreshTime] = useState<Date | null>(null);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 获取系统数据
  const fetchSystemData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await dataApi.getStatus();
      setSystemStatus(data);
      setLastRefreshTime(new Date());
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : '获取系统数据失败';
      setError(errorMessage);
      console.error('Failed to fetch system data:', err);
    } finally {
      setLoading(false);
    }
  }, [setSystemStatus]);

  // 自动刷新控制
  const setupAutoRefresh = useCallback(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchSystemData, 30000); // 30秒刷新一次
      setRefreshInterval(interval);
      return interval;
    }
    return null;
  }, [autoRefresh, fetchSystemData]);

  // 清理定时器
  const clearAutoRefresh = useCallback(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  }, [refreshInterval]);

  // 初始化数据加载
  useEffect(() => {
    fetchSystemData();
  }, [fetchSystemData]);

  // 自动刷新控制
  useEffect(() => {
    const interval = setupAutoRefresh();
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [setupAutoRefresh]);

  // 页面清理
  useEffect(() => {
    return () => {
      clearAutoRefresh();
    };
  }, [clearAutoRefresh]);

  // 手动刷新
  const handleManualRefresh = useCallback(() => {
    fetchSystemData();
  }, [fetchSystemData]);

  // 切换自动刷新
  const handleAutoRefreshToggle = (checked: boolean) => {
    setAutoRefresh(checked);
    if (!checked) {
      clearAutoRefresh();
    } else {
      setupAutoRefresh();
    }
  };

  // 计算系统统计数据
  const getSystemStats = () => {
    if (!systemStatus) {
      return {
        totalSymbols: 0,
        totalSize: 0,
        availableDataSources: 0,
        lastSync: undefined,
      };
    }

    const totalSymbols = systemStatus.storage_info.reduce(
      (sum, info) => sum + info.symbols_count,
      0
    );

    const totalSize = systemStatus.storage_info.reduce(
      (sum, info) => sum + info.total_size_mb,
      0
    );

    const availableDataSources = systemStatus.data_sources.filter(
      source => source.enabled && source.status === 'available'
    ).length;

    return {
      totalSymbols,
      totalSize,
      availableDataSources,
      lastSync: systemStatus.last_sync,
    };
  };

  const stats = getSystemStats();

  return (
    <div style={{ padding: '24px', minHeight: '100vh' }}>
      {/* 页面头部 */}
      <div style={{ marginBottom: '24px' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '16px',
          }}
        >
          <Title level={2} style={{ margin: 0 }}>
            AI量化系统 - 数据概览
          </Title>

          <Space>
            <Tooltip title="演示模式开关">
              <Switch
                checked={isDemoMode}
                onChange={setIsDemoMode}
                checkedChildren="演示"
                unCheckedChildren="实时"
              />
            </Tooltip>

            <Tooltip title="自动刷新(30秒)">
              <Switch
                checked={autoRefresh}
                onChange={handleAutoRefreshToggle}
                checkedChildren={<PlayCircleOutlined />}
                unCheckedChildren={<PauseCircleOutlined />}
              />
            </Tooltip>

            <Button
              type="primary"
              icon={<ReloadOutlined />}
              onClick={handleManualRefresh}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        </div>

        {/* 状态信息栏 */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Space>
            {error && (
              <Alert
                message={error}
                type="error"
                showIcon
                style={{ marginRight: '16px' }}
              />
            )}

            {lastRefreshTime && (
              <span style={{ color: '#666', fontSize: '12px' }}>
                最后更新: {lastRefreshTime.toLocaleTimeString('zh-CN')}
              </span>
            )}
          </Space>

          <Space>
            <span style={{ color: '#666', fontSize: '12px' }}>
              {notifications.length} 条通知
            </span>

            {systemStatus && (
              <span style={{ color: '#52c41a', fontSize: '12px' }}>
                ● 系统运行正常
              </span>
            )}
          </Space>
        </div>
      </div>

      {/* 系统状态概览卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={8}>
          <Card>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <DatabaseOutlined
                style={{
                  fontSize: '24px',
                  color: '#1890ff',
                  marginRight: '12px',
                }}
              />
              <div>
                <div
                  style={{
                    fontSize: '24px',
                    fontWeight: 'bold',
                    color: '#1890ff',
                  }}
                >
                  {stats.totalSymbols}
                </div>
                <div style={{ fontSize: '14px', color: '#666' }}>股票总数</div>
              </div>
            </div>
          </Card>
        </Col>

        <Col xs={24} sm={8}>
          <Card>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <CloudServerOutlined
                style={{
                  fontSize: '24px',
                  color: '#52c41a',
                  marginRight: '12px',
                }}
              />
              <div>
                <div
                  style={{
                    fontSize: '24px',
                    fontWeight: 'bold',
                    color: '#52c41a',
                  }}
                >
                  {(stats.totalSize / 1024).toFixed(1)}
                </div>
                <div style={{ fontSize: '14px', color: '#666' }}>
                  数据容量(GB)
                </div>
              </div>
            </div>
          </Card>
        </Col>

        <Col xs={24} sm={8}>
          <Card>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <LineChartOutlined
                style={{
                  fontSize: '24px',
                  color: '#722ed1',
                  marginRight: '12px',
                }}
              />
              <div>
                <div
                  style={{
                    fontSize: '24px',
                    fontWeight: 'bold',
                    color: '#722ed1',
                  }}
                >
                  {stats.availableDataSources}
                </div>
                <div style={{ fontSize: '14px', color: '#666' }}>
                  可用数据源
                </div>
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 详细状态卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={8}>
          {systemStatus && (
            <DataSourceStatusCard
              title="数据源状态"
              dataSources={systemStatus.data_sources}
              loading={loading}
            />
          )}
        </Col>

        <Col xs={24} lg={8}>
          {systemStatus && (
            <StorageStatusCard
              title="存储状态"
              storageInfo={systemStatus.storage_info}
              loading={loading}
            />
          )}
        </Col>

        <Col xs={24} lg={8}>
          <SystemStatsCard title="系统统计" stats={stats} loading={loading} />
        </Col>
      </Row>

      {/* 快速操作区 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24}>
          <QuickActions onRefreshData={handleManualRefresh} />
        </Col>
      </Row>

      {/* 系统概览图表 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24}>
          <Card title="系统概览图表" size="small">
            <SystemOverview theme="light" />
          </Card>
        </Col>
      </Row>

      {/* 演示模式内容 */}
      {isDemoMode && (
        <>
          <Divider>
            <span style={{ color: '#666' }}>演示内容</span>
          </Divider>

          {/* API测试组件 */}
          <div style={{ marginBottom: '24px' }}>
            <ApiTest />
          </div>

          {/* ECharts图表演示 */}
          <div style={{ marginBottom: '24px' }}>
            <ChartDemo />
          </div>
        </>
      )}

      {/* 开发进度说明 */}
      <Card title="开发进度" size="small">
        <div>
          <Title level={4}>✅ 已完成功能</Title>
          <ul>
            <li>✅ React+TypeScript+Vite 项目脚手架</li>
            <li>✅ Ant Design UI框架集成</li>
            <li>✅ API服务层和状态管理（Zustand）</li>
            <li>✅ 基础路由和页面框架</li>
            <li>✅ 与后端API的接口调用测试</li>
            <li>✅ ECharts图表组件库（K线图、折线图、柱状图）</li>
            <li>✅ 专业数据概览Dashboard页面</li>
          </ul>

          <Title level={4}>🔄 待开发功能</Title>
          <ul>
            <li>🔄 回测工作台页面</li>
            <li>🔄 WebSocket实时通信</li>
            <li>🔄 策略管理页面</li>
            <li>🔄 AI助手和纸上交易</li>
            <li>🔄 系统集成测试与性能优化</li>
          </ul>

          <Title level={4}>📊 系统特性</Title>
          <ul>
            <li>响应式设计，支持多屏幕尺寸</li>
            <li>自动数据刷新（30秒间隔）</li>
            <li>实时系统状态监控</li>
            <li>专业金融数据可视化</li>
            <li>模块化组件设计</li>
            <li>TypeScript类型安全保障</li>
          </ul>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;
