import { Spin, Card, Space, Typography } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface LoadingPageProps {
  tip?: string;
  size?: 'small' | 'default' | 'large';
  spinning?: boolean;
  style?: React.CSSProperties;
  className?: string;
}

/**
 * 页面加载状态组件
 * 提供统一的加载UI和自定义加载提示
 */
const LoadingPage: React.FC<LoadingPageProps> = ({
  tip = '加载中...',
  size = 'large',
  spinning = true,
  style,
  className,
}) => {
  return (
    <div
      className={className}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '300px',
        width: '100%',
        ...style,
      }}
    >
      <Card
        bordered={false}
        style={{
          background: 'transparent',
          textAlign: 'center',
          boxShadow: 'none',
        }}
      >
        <Space direction="vertical" size="large">
          <Spin
            spinning={spinning}
            size={size}
            indicator={
              <LoadingOutlined
                style={{ fontSize: size === 'large' ? 48 : 24 }}
                spin
              />
            }
          />

          {tip && (
            <Text type="secondary" style={{ fontSize: '14px' }}>
              {tip}
            </Text>
          )}
        </Space>
      </Card>
    </div>
  );
};

/**
 * 全屏加载组件
 */
export const FullScreenLoading: React.FC<
  Omit<LoadingPageProps, 'style'>
> = props => (
  <LoadingPage
    {...props}
    style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
      zIndex: 9999,
      minHeight: '100vh',
    }}
  />
);

/**
 * 内嵌加载组件
 */
export const InlineLoading: React.FC<LoadingPageProps> = props => (
  <LoadingPage
    {...props}
    size="default"
    style={{
      minHeight: '200px',
      ...props.style,
    }}
  />
);

export default LoadingPage;

