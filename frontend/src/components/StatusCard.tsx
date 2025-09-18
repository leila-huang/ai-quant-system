import { Card, Statistic, Progress, Tag, Space, Typography } from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  SyncOutlined,
} from '@ant-design/icons';
import type { DataSourceInfo, StorageInfo } from '@/types/api';

const { Text } = Typography;

interface StatusCardProps {
  title: string;
  loading?: boolean;
  className?: string;
}

interface DataSourceStatusProps extends StatusCardProps {
  dataSources: DataSourceInfo[];
}

interface StorageStatusProps extends StatusCardProps {
  storageInfo: StorageInfo[];
}

interface SystemStatsProps extends StatusCardProps {
  stats: {
    totalSymbols: number;
    totalSize: number;
    availableDataSources: number;
    lastSync?: string;
  };
}

// 数据源状态卡片
export const DataSourceStatusCard: React.FC<DataSourceStatusProps> = ({
  title,
  dataSources,
  loading = false,
  className,
}) => {
  const getStatusIcon = (enabled: boolean, status: string) => {
    if (!enabled) {
      return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
    }

    switch (status.toLowerCase()) {
      case 'available':
      case 'connected':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'connecting':
      case 'syncing':
        return <SyncOutlined spin style={{ color: '#1890ff' }} />;
      default:
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
    }
  };

  const getStatusColor = (enabled: boolean, status: string) => {
    if (!enabled) return 'error';

    switch (status.toLowerCase()) {
      case 'available':
      case 'connected':
        return 'success';
      case 'connecting':
      case 'syncing':
        return 'processing';
      default:
        return 'warning';
    }
  };

  const enabledCount = dataSources.filter(source => source.enabled).length;
  const availableCount = dataSources.filter(
    source => source.enabled && source.status.toLowerCase() === 'available'
  ).length;

  return (
    <Card title={title} loading={loading} className={className} size="small">
      <div style={{ marginBottom: '16px' }}>
        <Statistic
          title="可用数据源"
          value={availableCount}
          suffix={`/ ${dataSources.length}`}
          valueStyle={{
            color: availableCount === enabledCount ? '#52c41a' : '#faad14',
          }}
        />
      </div>

      <Space direction="vertical" style={{ width: '100%' }} size="small">
        {dataSources.map((source, index) => (
          <div
            key={index}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px',
              backgroundColor: '#fafafa',
              borderRadius: '4px',
            }}
          >
            <Space>
              {getStatusIcon(source.enabled, source.status)}
              <Text strong>{source.name}</Text>
            </Space>
            <Tag color={getStatusColor(source.enabled, source.status) as any}>
              {source.enabled ? source.status : 'disabled'}
            </Tag>
          </div>
        ))}
      </Space>

      {dataSources.length > 0 && dataSources[0].last_update && (
        <div style={{ marginTop: '12px', textAlign: 'center' }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            最后更新:{' '}
            {new Date(dataSources[0].last_update).toLocaleString('zh-CN')}
          </Text>
        </div>
      )}
    </Card>
  );
};

// 存储状态卡片
export const StorageStatusCard: React.FC<StorageStatusProps> = ({
  title,
  storageInfo,
  loading = false,
  className,
}) => {
  const totalSize = storageInfo.reduce(
    (sum, info) => sum + info.total_size_mb,
    0
  );
  const totalFiles = storageInfo.reduce(
    (sum, info) => sum + info.total_files,
    0
  );
  const totalSymbols = storageInfo.reduce(
    (sum, info) => sum + info.symbols_count,
    0
  );

  const formatSize = (mb: number) => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <Card title={title} loading={loading} className={className} size="small">
      <div style={{ marginBottom: '16px' }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Statistic
            title="存储总量"
            value={formatSize(totalSize)}
            valueStyle={{ fontSize: '20px', fontWeight: 'bold' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <Statistic
              title="文件数"
              value={totalFiles}
              valueStyle={{ fontSize: '16px' }}
            />
            <Statistic
              title="股票数"
              value={totalSymbols}
              valueStyle={{ fontSize: '16px' }}
            />
          </div>
        </Space>
      </div>

      <Space direction="vertical" style={{ width: '100%' }} size="small">
        {storageInfo.map((storage, index) => (
          <div
            key={index}
            style={{
              padding: '8px',
              backgroundColor: '#fafafa',
              borderRadius: '4px',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: '4px',
              }}
            >
              <Text strong>{storage.type.toUpperCase()}</Text>
              <Text type="secondary">{formatSize(storage.total_size_mb)}</Text>
            </div>
            <div style={{ fontSize: '12px', color: '#666' }}>
              {storage.symbols_count} 支股票, {storage.total_files} 个文件
            </div>
            <Progress
              percent={(storage.total_size_mb / totalSize) * 100}
              size="small"
              showInfo={false}
              style={{ marginTop: '4px' }}
            />
          </div>
        ))}
      </Space>
    </Card>
  );
};

// 系统统计卡片
export const SystemStatsCard: React.FC<SystemStatsProps> = ({
  title,
  stats,
  loading = false,
  className,
}) => {
  const formatSize = (mb: number) => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(0)} MB`;
  };

  return (
    <Card title={title} loading={loading} className={className} size="small">
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '16px',
          }}
        >
          <Statistic
            title="总股票数"
            value={stats.totalSymbols}
            suffix="支"
            valueStyle={{ color: '#1890ff' }}
          />
          <Statistic
            title="数据容量"
            value={formatSize(stats.totalSize)}
            valueStyle={{ color: '#52c41a' }}
          />
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '16px',
          }}
        >
          <Statistic
            title="数据源"
            value={stats.availableDataSources}
            suffix="个"
            valueStyle={{ color: '#722ed1' }}
          />
          <div>
            <div
              style={{ fontSize: '14px', color: '#666', marginBottom: '4px' }}
            >
              系统状态
            </div>
            <Tag color="success" icon={<CheckCircleOutlined />}>
              运行正常
            </Tag>
          </div>
        </div>

        {stats.lastSync && (
          <div
            style={{
              textAlign: 'center',
              padding: '8px',
              backgroundColor: '#f6f8fa',
              borderRadius: '4px',
            }}
          >
            <Text type="secondary" style={{ fontSize: '12px' }}>
              最后同步: {new Date(stats.lastSync).toLocaleString('zh-CN')}
            </Text>
          </div>
        )}
      </Space>
    </Card>
  );
};

export default {
  DataSourceStatusCard,
  StorageStatusCard,
  SystemStatsCard,
};

