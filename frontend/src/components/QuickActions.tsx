import { Card, Button, Space, App } from 'antd';
import {
  SyncOutlined,
  DatabaseOutlined,
  BarChartOutlined,
  SettingOutlined,
  ReloadOutlined,
  CloudDownloadOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useState } from 'react';
import { dataApi, dataSyncApi, apiRequest } from '@/services';
import type {
  DetailedHealth,
  HealthMetrics,
  DatabaseHealth,
} from '@/types/api';
import { pollUntil } from '@/utils/polling';
import { useAppStore } from '@/stores';

interface QuickActionsProps {
  onRefreshData?: () => void;
  className?: string;
}

/**
 * 快速操作区组件
 * 提供常用的系统操作快捷入口
 */
const QuickActions: React.FC<QuickActionsProps> = ({
  onRefreshData,
  className,
}) => {
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const { message, modal } = App.useApp();
  const { addNotification } = useAppStore();

  // 设置加载状态
  const setActionLoading = (action: string, isLoading: boolean) => {
    setLoading(prev => ({ ...prev, [action]: isLoading }));
  };

  // 刷新系统数据
  const handleRefreshData = async () => {
    setActionLoading('refresh', true);

    try {
      message.loading({ content: '正在刷新数据...', key: 'refresh' });

      // 并行获取数据状态与存储统计
      const [status] = await Promise.all([
        dataApi.getStatus(),
        // 可按需展示：await dataApi.getStorageStats()
      ]);

      message.success({
        content: `数据刷新完成（数据源:${status.data_sources?.[0]?.status || 'unknown'}）`,
        key: 'refresh',
        duration: 2,
      });

      addNotification({
        type: 'success',
        title: '刷新成功',
        message: '系统数据已更新到最新状态',
      });

      // 通知父组件刷新
      if (onRefreshData) {
        onRefreshData();
      }
    } catch {
      message.error({ content: '刷新失败', key: 'refresh' });

      addNotification({
        type: 'error',
        title: '刷新失败',
        message: '系统数据刷新时发生错误',
      });
    } finally {
      setActionLoading('refresh', false);
    }
  };

  // 数据同步
  const handleDataSync = () => {
    modal.confirm({
      title: '确认数据同步',
      icon: <ExclamationCircleOutlined />,
      content: '数据同步可能需要较长时间，是否继续？',
      okText: '开始同步',
      cancelText: '取消',
      onOk: async () => {
        setActionLoading('sync', true);

        try {
          message.loading({ content: '正在创建同步任务...', key: 'sync' });

          // 创建异步数据同步任务（默认AKShare最近30天）
          const resp = await dataSyncApi.createSyncTask({
            async_mode: true,
            data_source: 'akshare',
          });
          const { task_id } = resp;

          const last = await pollUntil({
            fetch: () => dataSyncApi.getSyncTaskStatus(task_id),
            isDone: s =>
              ['success', 'failed', 'cancelled'].includes(s.status as string),
            intervalMs: 2000,
            timeoutMs: 120000,
            onTick: s => {
              const progressText = `同步进行中：${Math.round(s.progress)}%`;
              message.loading({ content: progressText, key: 'sync' });
            },
          });

          if (last.status === 'success') {
            message.success({
              content: '数据同步完成',
              key: 'sync',
              duration: 3,
            });
            addNotification({
              type: 'success',
              title: '同步完成',
              message: `已成功同步股票数据（共${last.symbols_completed || 0}只）`,
            });
            if (onRefreshData) onRefreshData();
          } else if (last.status === 'running' || last.status === 'pending') {
            message.warning({
              content: '同步仍在进行中，稍后可在系统监控查看进度',
              key: 'sync',
              duration: 3,
            });
          } else {
            message.error({
              content: `同步结束：${last.status}`,
              key: 'sync',
              duration: 3,
            });
            addNotification({
              type: 'error',
              title: '同步未成功',
              message: last.error_message || '请稍后重试',
            });
          }
        } catch {
          message.error({ content: '同步失败', key: 'sync' });

          addNotification({
            type: 'error',
            title: '同步失败',
            message: '数据同步过程中发生错误',
          });
        } finally {
          setActionLoading('sync', false);
        }
      },
    });
  };

  // 获取样本数据
  const handleGetSampleData = async () => {
    setActionLoading('sample', true);

    try {
      message.loading({ content: '正在创建样本数据...', key: 'sample' });

      const result = await dataApi.createSampleData('SAMPLE001');

      message.success({
        content: `样本数据创建成功: ${result.symbol}`,
        key: 'sample',
        duration: 3,
      });

      addNotification({
        type: 'success',
        title: '样本数据创建成功',
        message: `已创建 ${result.symbol} 的测试数据，包含 ${result.bars_count} 个数据点`,
      });

      // 刷新页面数据
      if (onRefreshData) {
        onRefreshData();
      }
    } catch {
      message.error({ content: '创建样本数据失败', key: 'sample' });

      addNotification({
        type: 'error',
        title: '创建失败',
        message: '样本数据创建时发生错误',
      });
    } finally {
      setActionLoading('sample', false);
    }
  };

  // 系统监控
  const handleSystemMonitor = async () => {
    setActionLoading('monitor', true);
    try {
      const promises: [Promise<DetailedHealth>, Promise<HealthMetrics>] = [
        apiRequest.get<DetailedHealth>('/health/detailed', undefined, {
          cache: false,
        }),
        apiRequest.get<HealthMetrics>('/health/metrics', undefined, {
          cache: false,
        }),
      ];
      const [health, metrics] = await Promise.all(promises);

      modal.info({
        title: '系统监控',
        content: (
          <div style={{ padding: '16px 0' }}>
            <p>总体状态：{health.status}</p>
            <p>
              版本：{health.version} 环境：{health.environment}
            </p>
            <p>数据库连接：{health.components?.database?.connection?.status}</p>
            <p>
              连接池：{health.components?.database?.pool?.status}（利用率{' '}
              {health.components?.database?.pool?.utilization_percent || 0}%）
            </p>
            <p>API平均响应：{metrics.api_metrics?.avg_response_time || 0} ms</p>
          </div>
        ),
        okText: '知道了',
      });
    } catch {
      message.error('获取系统监控信息失败');
    } finally {
      setActionLoading('monitor', false);
    }
  };

  // 系统设置
  const handleSystemSettings = () => {
    modal.info({
      title: '系统设置',
      content: (
        <div style={{ padding: '16px 0' }}>
          <p>系统设置功能将在P3阶段实现，敬请期待！</p>
          <p>计划功能包括：</p>
          <ul style={{ marginLeft: '20px' }}>
            <li>数据源配置</li>
            <li>系统参数设置</li>
            <li>用户权限管理</li>
            <li>主题和界面设置</li>
          </ul>
        </div>
      ),
      okText: '知道了',
    });
  };

  const actions = [
    {
      key: 'refresh',
      title: '刷新数据',
      icon: <ReloadOutlined />,
      description: '刷新系统状态和统计信息',
      onClick: handleRefreshData,
      type: 'primary' as const,
    },
    {
      key: 'sync',
      title: '数据同步',
      icon: <SyncOutlined />,
      description: '从数据源同步最新数据',
      onClick: handleDataSync,
      type: 'default' as const,
    },
    {
      key: 'sample',
      title: '样本数据',
      icon: <CloudDownloadOutlined />,
      description: '创建测试用样本数据',
      onClick: handleGetSampleData,
      type: 'default' as const,
    },
    {
      key: 'database',
      title: '数据库',
      icon: <DatabaseOutlined />,
      description: '数据库管理和维护',
      onClick: async () => {
        setActionLoading('database', true);
        try {
          const db = await apiRequest.get<DatabaseHealth>(
            '/health/database',
            undefined,
            { cache: false }
          );
          modal.info({
            title: '数据库健康',
            content: (
              <div style={{ padding: '16px 0' }}>
                <p>连接：{db.connection?.status}</p>
                <p>
                  连接池：{db.connection_pool?.status}（已借出{' '}
                  {db.connection_pool?.checked_out || 0}/
                  {db.connection_pool?.pool_size || 0}）
                </p>
                <p>
                  平均查询耗时：{db.performance?.average_query_time_ms || 0} ms
                </p>
                <p>总体：{db.overall_status}</p>
              </div>
            ),
            okText: '知道了',
          });
        } catch {
          message.error('获取数据库健康信息失败');
        } finally {
          setActionLoading('database', false);
        }
      },
      type: 'default' as const,
    },
    {
      key: 'monitor',
      title: '系统监控',
      icon: <BarChartOutlined />,
      description: '查看系统运行状态',
      onClick: handleSystemMonitor,
      type: 'default' as const,
    },
    {
      key: 'settings',
      title: '系统设置',
      icon: <SettingOutlined />,
      description: '配置系统参数',
      onClick: handleSystemSettings,
      type: 'default' as const,
    },
  ];

  return (
    <Card title="快速操作" size="small" className={className}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '12px',
        }}
      >
        {actions.map(action => (
          <Button
            key={action.key}
            type={action.type}
            icon={action.icon}
            loading={loading[action.key]}
            onClick={action.onClick}
            style={{
              height: 'auto',
              padding: '12px 8px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
            }}
          >
            <div style={{ marginBottom: '4px' }}>{action.title}</div>
            <div
              style={{
                fontSize: '11px',
                color: '#666',
                lineHeight: 1.2,
                height: '26px',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              {action.description}
            </div>
          </Button>
        ))}
      </div>

      <div
        style={{
          marginTop: '16px',
          padding: '8px',
          backgroundColor: '#f6f8fa',
          borderRadius: '4px',
          textAlign: 'center',
        }}
      >
        <Space size="large">
          <span style={{ fontSize: '12px', color: '#666' }}>
            快捷键: Ctrl+R 刷新
          </span>
          <span style={{ fontSize: '12px', color: '#666' }}>Ctrl+S 同步</span>
        </Space>
      </div>
    </Card>
  );
};

export default QuickActions;
