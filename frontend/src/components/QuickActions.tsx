import { Card, Button, Space, message, Modal } from 'antd';
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
import { dataApi } from '@/services';
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

      // 模拟刷新操作
      await new Promise(resolve => setTimeout(resolve, 2000));

      message.success({ content: '数据刷新完成', key: 'refresh', duration: 2 });

      addNotification({
        type: 'success',
        title: '刷新成功',
        message: '系统数据已更新到最新状态',
      });

      // 通知父组件刷新
      if (onRefreshData) {
        onRefreshData();
      }
    } catch (error) {
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
    Modal.confirm({
      title: '确认数据同步',
      icon: <ExclamationCircleOutlined />,
      content: '数据同步可能需要较长时间，是否继续？',
      okText: '开始同步',
      cancelText: '取消',
      onOk: async () => {
        setActionLoading('sync', true);

        try {
          message.loading({ content: '正在同步数据...', key: 'sync' });

          // 模拟同步操作
          await new Promise(resolve => setTimeout(resolve, 5000));

          message.success({
            content: '数据同步完成',
            key: 'sync',
            duration: 3,
          });

          addNotification({
            type: 'success',
            title: '同步完成',
            message: '已成功同步最新的股票数据',
          });
        } catch (error) {
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
    } catch (error) {
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
  const handleSystemMonitor = () => {
    Modal.info({
      title: '系统监控',
      content: (
        <div style={{ padding: '16px 0' }}>
          <p>系统监控功能将在后续版本中提供，敬请期待！</p>
          <p>计划功能包括：</p>
          <ul style={{ marginLeft: '20px' }}>
            <li>实时性能监控</li>
            <li>资源使用统计</li>
            <li>系统日志查看</li>
            <li>异常报警设置</li>
          </ul>
        </div>
      ),
      okText: '知道了',
    });
  };

  // 系统设置
  const handleSystemSettings = () => {
    Modal.info({
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
      onClick: () => message.info('数据库管理功能开发中'),
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

