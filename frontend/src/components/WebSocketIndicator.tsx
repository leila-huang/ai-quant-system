import { useState, type ReactNode } from 'react';
import {
  Badge,
  // Tooltip, - unused
  Button,
  // Dropdown, - unused
  Space,
  Typography,
  Card,
  List,
  Tag,
  Popover,
} from 'antd';
import type { BadgeProps } from 'antd';
import {
  WifiOutlined,
  DisconnectOutlined,
  SyncOutlined,
  ExclamationCircleOutlined,
  BellOutlined,
  CloseOutlined,
} from '@ant-design/icons';
import {
  useWebSocketStore,
  useWebSocketNotifications,
  websocketSelectors,
} from '@/stores/websocketStore';
import type { WebSocketStatus, WebSocketMessage } from '@/services/websocket';

const { Text } = Typography;

interface WebSocketIndicatorProps {
  showNotifications?: boolean;
  className?: string;
}

/**
 * WebSocket连接状态指示器组件
 * 显示连接状态、消息统计、实时通知等
 */
const WebSocketIndicator: React.FC<WebSocketIndicatorProps> = ({
  showNotifications = true,
  className,
}) => {
  const [notificationsVisible, setNotificationsVisible] = useState(false);

  // WebSocket状态 - 使用更稳定的选择器方式
  const status = useWebSocketStore(websocketSelectors.status);
  const isConnected = useWebSocketStore(websocketSelectors.isConnected);
  const messageCount = useWebSocketStore(websocketSelectors.messageCount);
  const lastConnectTime = useWebSocketStore(websocketSelectors.lastConnectTime);
  const lastMessageTime = useWebSocketStore(websocketSelectors.lastMessageTime);

  // WebSocket操作 - 直接从state获取，避免不必要的重新渲染
  const connect = useWebSocketStore(state => state.connect);
  const disconnect = useWebSocketStore(state => state.disconnect);

  // 通知管理
  const { notifications, removeNotification, clearNotifications } =
    useWebSocketNotifications(20);

  // 获取状态图标和颜色
  const getStatusConfig = (
    status: WebSocketStatus
  ): {
    icon: ReactNode;
    color: BadgeProps['status'];
    text: string;
    description: string;
  } => {
    switch (status) {
      case 'connected':
        return {
          icon: <WifiOutlined />,
          color: 'success',
          text: '已连接',
          description: '实时连接正常',
        };
      case 'connecting':
        return {
          icon: <SyncOutlined spin />,
          color: 'processing',
          text: '连接中',
          description: '正在建立连接...',
        };
      case 'reconnecting':
        return {
          icon: <SyncOutlined spin />,
          color: 'warning',
          text: '重连中',
          description: '正在尝试重新连接...',
        };
      case 'disconnected':
        return {
          icon: <DisconnectOutlined />,
          color: 'default',
          text: '已断开',
          description: '连接已断开',
        };
      case 'failed':
        return {
          icon: <ExclamationCircleOutlined />,
          color: 'error',
          text: '连接失败',
          description: '连接失败，请检查网络',
        };
      default:
        return {
          icon: <DisconnectOutlined />,
          color: 'default',
          text: '未知',
          description: '状态未知',
        };
    }
  };

  const statusConfig = getStatusConfig(status);

  // 格式化时间
  const formatTime = (timeString?: string) => {
    if (!timeString) return '--';
    return new Date(timeString).toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  // 渲染连接详情
  const renderConnectionDetails = () => (
    <Card size="small" style={{ width: 280 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <div>
          <Text strong>连接状态</Text>
          <div style={{ marginTop: '4px' }}>
            <Space>
              <Badge status={statusConfig.color} />
              <Text>{statusConfig.text}</Text>
            </Space>
            <div style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              {statusConfig.description}
            </div>
          </div>
        </div>

        <div>
          <Text strong>统计信息</Text>
          <div style={{ marginTop: '4px', fontSize: '12px' }}>
            <div>消息总数: {messageCount}</div>
            <div>最后连接: {formatTime(lastConnectTime)}</div>
            <div>最后消息: {formatTime(lastMessageTime)}</div>
          </div>
        </div>

        <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '8px' }}>
          <Space>
            {!isConnected ? (
              <Button
                type="primary"
                size="small"
                onClick={connect}
                loading={status === 'connecting' || status === 'reconnecting'}
              >
                连接
              </Button>
            ) : (
              <Button size="small" onClick={disconnect}>
                断开
              </Button>
            )}

            <Button size="small" onClick={() => window.location.reload()}>
              刷新页面
            </Button>
          </Space>
        </div>
      </Space>
    </Card>
  );

  // 渲染通知列表
  const renderNotifications = () => {
    if (!showNotifications || notifications.length === 0) {
      return (
        <Card size="small" style={{ width: 300 }}>
          <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>
            暂无实时通知
          </div>
        </Card>
      );
    }

    return (
      <Card
        size="small"
        style={{ width: 300, maxHeight: 400, overflowY: 'auto' }}
        title={
          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
            <span>实时通知 ({notifications.length})</span>
            <Button type="text" size="small" onClick={clearNotifications}>
              清空
            </Button>
          </Space>
        }
      >
        <List
          size="small"
          dataSource={notifications.slice().reverse()} // 最新的在前
          renderItem={(notification, index) => (
            <List.Item
              style={{ padding: '8px 0' }}
              actions={[
                <Button
                  key="remove"
                  type="text"
                  size="small"
                  icon={<CloseOutlined />}
                  onClick={() =>
                    removeNotification(notifications.length - 1 - index)
                  }
                />,
              ]}
            >
              <List.Item.Meta
                title={
                  <Space>
                    <Tag
                      color={getMessageTypeColor(notification.type)}
                      style={{ fontSize: '10px' }}
                    >
                      {getMessageTypeText(notification.type)}
                    </Tag>
                    <Text style={{ fontSize: '12px' }}>
                      {getNotificationTitle(notification)}
                    </Text>
                  </Space>
                }
                description={
                  <div style={{ fontSize: '11px', color: '#666' }}>
                    {getNotificationDescription(notification)}
                    <div style={{ marginTop: '2px' }}>
                      {notification.timestamp &&
                        formatTime(notification.timestamp)}
                    </div>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Card>
    );
  };

  // 获取消息类型颜色
  const getMessageTypeColor = (type: string) => {
    switch (type) {
      case 'backtest_progress':
        return 'blue';
      case 'system_status':
        return 'orange';
      case 'data_sync':
        return 'green';
      case 'notification':
        return 'purple';
      default:
        return 'default';
    }
  };

  // 获取消息类型文本
  const getMessageTypeText = (type: string) => {
    switch (type) {
      case 'backtest_progress':
        return '回测';
      case 'system_status':
        return '系统';
      case 'data_sync':
        return '数据';
      case 'notification':
        return '通知';
      default:
        return type;
    }
  };

  // 获取通知标题
  const getNotificationTitle = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'backtest_progress':
        return `回测进度: ${message.data.progress * 100}%`;
      case 'system_status':
        return `系统状态: ${message.data.component}`;
      case 'data_sync':
        return `数据同步: ${message.data.source}`;
      default:
        return message.type;
    }
  };

  // 获取通知描述
  const getNotificationDescription = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'backtest_progress':
        return `${message.data.current_step} - ${message.data.status}`;
      case 'system_status':
        return `${message.data.status}: ${message.data.message}`;
      case 'data_sync':
        return `${message.data.action}: ${message.data.message || ''}`;
      default:
        return JSON.stringify(message.data);
    }
  };

  return (
    <Space className={className}>
      {/* 连接状态指示器 */}
      <Popover
        content={renderConnectionDetails()}
        title="WebSocket连接"
        trigger="hover"
        placement="bottomRight"
      >
        <Badge status={statusConfig.color} style={{ cursor: 'pointer' }}>
          <Space>
            {statusConfig.icon}
            <span style={{ fontSize: '12px' }}>{statusConfig.text}</span>
          </Space>
        </Badge>
      </Popover>

      {/* 实时通知指示器 */}
      {showNotifications && (
        <Popover
          content={renderNotifications()}
          title={null}
          trigger="click"
          open={notificationsVisible}
          onOpenChange={setNotificationsVisible}
          placement="bottomRight"
        >
          <Badge
            count={notifications.length}
            size="small"
            style={{ cursor: 'pointer' }}
          >
            <Button
              type="text"
              size="small"
              icon={<BellOutlined />}
              style={{ color: notifications.length > 0 ? '#1890ff' : '#999' }}
            />
          </Badge>
        </Popover>
      )}
    </Space>
  );
};

export default WebSocketIndicator;
