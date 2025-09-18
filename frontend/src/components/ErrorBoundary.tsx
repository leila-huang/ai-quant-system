import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Result, Button, Card, Typography, Space, Collapse, Tag } from 'antd';
import {
  ExclamationCircleOutlined,
  ReloadOutlined,
  BugOutlined,
  HomeOutlined,
} from '@ant-design/icons';

const { Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * 错误边界组件
 * 捕获子组件中的JavaScript错误，记录错误并显示错误UI
 */
class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    // 更新state，使下一次渲染显示错误UI
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    this.setState({
      error,
      errorInfo,
    });

    // 调用错误回调
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // 在开发环境中记录详细错误信息
    if (import.meta.env.DEV) {
      console.group('🚨 Error Boundary Details');
      console.error('Error:', error);
      console.error('Error Info:', errorInfo);
      console.error('Component Stack:', errorInfo.componentStack);
      console.groupEnd();
    }
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  private handleGoHome = () => {
    window.location.href = '/';
  };

  public render() {
    if (this.state.hasError) {
      // 如果提供了自定义fallback，使用它
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo } = this.state;
      const isDev = import.meta.env.DEV;

      return (
        <div
          style={{
            padding: '24px',
            minHeight: '400px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Card style={{ maxWidth: '800px', width: '100%' }}>
            <Result
              status="error"
              icon={<ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />}
              title="页面出现错误"
              subTitle={
                <Space direction="vertical" style={{ textAlign: 'left' }}>
                  <Text>
                    抱歉，页面遇到了意外错误。请尝试刷新页面或联系技术支持。
                  </Text>
                  {error && (
                    <Tag color="red" style={{ marginTop: '8px' }}>
                      错误类型: {error.name}
                    </Tag>
                  )}
                </Space>
              }
              extra={
                <Space>
                  <Button
                    type="primary"
                    icon={<ReloadOutlined />}
                    onClick={this.handleReload}
                  >
                    刷新页面
                  </Button>
                  <Button icon={<HomeOutlined />} onClick={this.handleGoHome}>
                    回到首页
                  </Button>
                  {isDev && (
                    <Button icon={<BugOutlined />} onClick={this.handleReset}>
                      重置错误
                    </Button>
                  )}
                </Space>
              }
            />

            {/* 开发环境显示详细错误信息 */}
            {isDev && (error || errorInfo) && (
              <Card
                title="错误详情 (仅开发环境显示)"
                size="small"
                style={{ marginTop: '16px' }}
              >
                <Collapse ghost>
                  {error && (
                    <Panel header="错误消息" key="error">
                      <Paragraph>
                        <Text code>{error.message}</Text>
                      </Paragraph>
                      {error.stack && (
                        <Paragraph>
                          <Text
                            code
                            style={{ fontSize: '12px', whiteSpace: 'pre-wrap' }}
                          >
                            {error.stack}
                          </Text>
                        </Paragraph>
                      )}
                    </Panel>
                  )}

                  {errorInfo && (
                    <Panel header="组件堆栈" key="componentStack">
                      <Paragraph>
                        <Text
                          code
                          style={{ fontSize: '12px', whiteSpace: 'pre-wrap' }}
                        >
                          {errorInfo.componentStack}
                        </Text>
                      </Paragraph>
                    </Panel>
                  )}
                </Collapse>
              </Card>
            )}
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * 页面级错误边界 Hook
 */
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  fallback?: ReactNode,
  onError?: (error: Error, errorInfo: ErrorInfo) => void
) => {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary fallback={fallback} onError={onError}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;

  return WrappedComponent;
};

export default ErrorBoundary;
