import { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  // Table, - unused
  Progress,
  Button,
  Space,
  // Typography, - unused
  Tag,
  Modal,
  Alert,
  Descriptions,
  Tabs,
  Divider,
} from 'antd';
import {
  // LineChartOutlined, - unused
  DownloadOutlined,
  EyeOutlined,
  SwapOutlined, // 替代CompareArrowsOutlined
} from '@ant-design/icons';
import { LineChart } from './Charts';
import type {
  BacktestResponse,
  BacktestStatusResponse,
  BacktestResultDetail,
  BacktestMetadata,
} from '@/types/api';
import type { TimeSeriesDataPoint } from '@/types/charts';
import { backtestApi } from '@/services';
import {
  useBacktestProgress,
  useWebSocketConnection,
} from '@/stores/websocketStore';
// formatChartValue unused, removed

// Text unused, removed
const { TabPane } = Tabs;

interface BacktestResultsProps {
  backtest: BacktestResponse | BacktestMetadata;
  onCompare?: (backtestId: string) => void;
  onGenerateReport?: (backtestId: string) => void;
  className?: string;
}

/**
 * 回测结果展示组件
 * 展示回测的各项性能指标、收益曲线、交易记录等
 */
const BacktestResults: React.FC<BacktestResultsProps> = ({
  backtest,
  onCompare,
  onGenerateReport,
  className,
}) => {
  const [detailVisible, setDetailVisible] = useState(false);
  const [resultDetail, setResultDetail] = useState<BacktestResultDetail | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<BacktestStatusResponse | null>(null);

  // 使用WebSocket获取实时回测进度
  const { connect } = useWebSocketConnection(true); // 自动连接
  const backtestProgress = useBacktestProgress(backtest.backtest_id);

  // 当WebSocket收到回测进度更新时，更新本地状态
  useEffect(() => {
    if (backtestProgress) {
      setStatus({
        backtest_id: backtest.backtest_id,
        status: backtestProgress.status,
        progress: backtestProgress.progress,
        current_step: backtestProgress.current_step,
        error_message: backtestProgress.error_message,
        started_at: '', // WebSocket消息中没有这些字段，保持为空
        updated_at: new Date().toISOString(),
      });
    }
  }, [backtestProgress, backtest.backtest_id]);

  // 确保WebSocket连接已建立
  useEffect(() => {
    connect();
  }, [connect]);

  // 获取详细结果
  const fetchDetailResults = async () => {
    setLoading(true);
    try {
      const detail = await backtestApi.getBacktestResults(backtest.backtest_id);
      setResultDetail(detail);
      setDetailVisible(true);
    } catch (error) {
      console.error('获取回测详细结果失败:', error);
      Modal.error({
        title: '获取失败',
        content: '无法获取回测详细结果，请稍后重试。',
      });
    } finally {
      setLoading(false);
    }
  };

  // 生成收益曲线数据
  const generateReturnsData = (): TimeSeriesDataPoint[] => {
    if (!resultDetail?.results.returns) return [];

    return Object.entries(resultDetail.results.returns).map(
      ([date, value]) => ({
        timestamp: date,
        value: Number(value) * 100, // 转换为百分比
      })
    );
  };

  // 获取状态标签颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'processing';
      case 'pending':
        return 'default';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  // 获取状态文本
  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed':
        return '已完成';
      case 'running':
        return '运行中';
      case 'pending':
        return '等待中';
      case 'failed':
        return '失败';
      default:
        return status;
    }
  };

  // 格式化数值
  const formatMetric = (
    value: number,
    type: 'percent' | 'number' | 'currency' = 'number'
  ) => {
    if (typeof value !== 'number' || isNaN(value)) return '--';

    switch (type) {
      case 'percent':
        return `${(value * 100).toFixed(2)}%`;
      case 'currency':
        return `¥${value.toLocaleString()}`;
      default:
        return value.toFixed(4);
    }
  };

  // 渲染性能指标卡片
  const renderPerformanceMetrics = () => {
    // metrics unused, removed
    // currentStatus unused, removed

    return (
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title="总收益率"
              value={formatMetric(backtest.total_return || 0, 'percent')}
              valueStyle={{
                color:
                  (backtest.total_return || 0) >= 0 ? '#3f8600' : '#cf1322',
                fontSize: '18px',
                fontWeight: 'bold',
              }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title="年化收益率"
              value={formatMetric(backtest.annual_return || 0, 'percent')}
              valueStyle={{
                color:
                  (backtest.annual_return || 0) >= 0 ? '#3f8600' : '#cf1322',
                fontSize: '18px',
                fontWeight: 'bold',
              }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title="最大回撤"
              value={formatMetric(
                Math.abs(backtest.max_drawdown || 0),
                'percent'
              )}
              valueStyle={{
                color: '#cf1322',
                fontSize: '18px',
                fontWeight: 'bold',
              }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title="夏普比率"
              value={formatMetric(backtest.sharpe_ratio || 0)}
              valueStyle={{
                color:
                  (backtest.sharpe_ratio || 0) >= 1 ? '#3f8600' : '#faad14',
                fontSize: '18px',
                fontWeight: 'bold',
              }}
            />
          </Card>
        </Col>
      </Row>
    );
  };

  // 渲染执行状态
  const renderExecutionStatus = () => {
    const currentStatus = status || backtest;

    if (currentStatus.status === 'completed') return null;

    return (
      <Alert
        message={
          <Space>
            <span>回测状态: {getStatusText(currentStatus.status)}</span>
            <Tag color={getStatusColor(currentStatus.status)}>
              {getStatusText(currentStatus.status)}
            </Tag>
          </Space>
        }
        description={
          <div>
            {status && (
              <>
                <div style={{ marginBottom: '8px' }}>
                  当前步骤: {status.current_step}
                </div>
                <Progress
                  percent={Math.round(status.progress * 100)}
                  size="small"
                  status={status.status === 'failed' ? 'exception' : 'active'}
                />
                {status.error_message && (
                  <div style={{ marginTop: '8px', color: '#cf1322' }}>
                    错误信息: {status.error_message}
                  </div>
                )}
              </>
            )}
          </div>
        }
        type={currentStatus.status === 'failed' ? 'error' : 'info'}
        style={{ marginBottom: '16px' }}
      />
    );
  };

  // 渲染详细结果Modal
  const renderDetailModal = () => {
    if (!resultDetail) return null;

    const returnsData = generateReturnsData();

    return (
      <Modal
        title={`回测详细结果 - ${backtest.backtest_name}`}
        open={detailVisible}
        onCancel={() => setDetailVisible(false)}
        width={1200}
        footer={[
          <Button key="close" onClick={() => setDetailVisible(false)}>
            关闭
          </Button>,
        ]}
      >
        <Tabs defaultActiveKey="overview">
          <TabPane tab="概览" key="overview">
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="回测ID">
                {resultDetail.backtest_id}
              </Descriptions.Item>
              <Descriptions.Item label="策略名称">
                {resultDetail.metadata.strategy_config?.strategy_name}
              </Descriptions.Item>
              <Descriptions.Item label="回测时间范围">
                {resultDetail.metadata.start_date} ~{' '}
                {resultDetail.metadata.end_date}
              </Descriptions.Item>
              <Descriptions.Item label="股票数量">
                {resultDetail.metadata.universe?.length || 0} 只
              </Descriptions.Item>
              <Descriptions.Item label="交易笔数">
                {resultDetail.results.trades?.length || 0}
              </Descriptions.Item>
              <Descriptions.Item label="执行时间">
                {resultDetail.metadata.execution_time_seconds?.toFixed(2)} 秒
              </Descriptions.Item>
            </Descriptions>
          </TabPane>

          <TabPane tab="收益曲线" key="returns">
            {returnsData.length > 0 && (
              <LineChart
                data={returnsData}
                title="策略收益曲线"
                xAxisLabel="时间"
                yAxisLabel="收益率(%)"
                height={400}
                smooth
                area
                color="#1890ff"
                formatter={value => `${value.toFixed(2)}%`}
              />
            )}
          </TabPane>

          <TabPane tab="详细指标" key="metrics">
            <Row gutter={[16, 16]}>
              {Object.entries(resultDetail.results.metrics || {}).map(
                ([key, value]) => (
                  <Col xs={12} sm={8} lg={6} key={key}>
                    <Statistic
                      title={key
                        .replace(/_/g, ' ')
                        .replace(/\b\w/g, l => l.toUpperCase())}
                      value={
                        typeof value === 'number'
                          ? value.toFixed(4)
                          : String(value)
                      }
                      valueStyle={{ fontSize: '14px' }}
                    />
                  </Col>
                )
              )}
            </Row>
          </TabPane>
        </Tabs>
      </Modal>
    );
  };

  return (
    <div className={className}>
      <Card
        title={
          <Space>
            <span>{backtest.backtest_name}</span>
            <Tag color={getStatusColor(backtest.status)}>
              {getStatusText(backtest.status)}
            </Tag>
          </Space>
        }
        size="small"
        extra={
          <Space>
            {backtest.status === 'completed' && (
              <>
                <Button
                  size="small"
                  icon={<EyeOutlined />}
                  onClick={fetchDetailResults}
                  loading={loading}
                >
                  查看详情
                </Button>

                {onCompare && (
                  <Button
                    size="small"
                    icon={<SwapOutlined />}
                    onClick={() => onCompare(backtest.backtest_id)}
                  >
                    对比
                  </Button>
                )}

                {onGenerateReport && (
                  <Button
                    size="small"
                    icon={<DownloadOutlined />}
                    onClick={() => onGenerateReport(backtest.backtest_id)}
                  >
                    生成报告
                  </Button>
                )}
              </>
            )}
          </Space>
        }
      >
        {renderExecutionStatus()}

        {backtest.status === 'completed' && (
          <>
            {renderPerformanceMetrics()}

            <Divider style={{ margin: '16px 0' }} />

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12}>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  <div>
                    回测时间:{' '}
                    {'started_at' in backtest ? backtest.started_at : ''}
                  </div>
                  <div>
                    执行时长: {backtest.execution_time_seconds?.toFixed(2)} 秒
                  </div>
                  <div>交易笔数: {backtest.trade_count} 笔</div>
                </div>
              </Col>

              <Col xs={24} sm={12}>
                <div
                  style={{
                    fontSize: '12px',
                    color: '#666',
                    textAlign: 'right',
                  }}
                >
                  <div>
                    策略类型:{' '}
                    {'strategy_config' in backtest
                      ? backtest.strategy_config?.strategy_type
                      : ''}
                  </div>
                  <div>
                    股票池:{' '}
                    {'universe' in backtest
                      ? `${backtest.universe?.length || 0} 只股票`
                      : ''}
                  </div>
                  <div>回测ID: {backtest.backtest_id}</div>
                </div>
              </Col>
            </Row>
          </>
        )}
      </Card>

      {renderDetailModal()}
    </div>
  );
};

export default BacktestResults;
