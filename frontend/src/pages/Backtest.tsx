import { useState, useCallback } from 'react';
import {
  // Row, Col - unused
  Card,
  Tabs,
  Button,
  Space,
  Typography,
  Modal,
  message,
  Alert,
  Divider,
} from 'antd';
import {
  PlayCircleOutlined,
  HistoryOutlined,
  // SwapOutlined, // 替代CompareArrowsOutlined - unused
  FileTextOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';
import BacktestForm from '@/components/BacktestForm';
import BacktestResults from '@/components/BacktestResults';
import BacktestHistory from '@/components/BacktestHistory';
import type { BacktestRequest, BacktestResponse } from '@/types/api';
import { backtestApi } from '@/services';
import { useAppStore } from '@/stores';

const { Title, Text } = Typography;
// TabPane unused, removed

/**
 * 回测工作台主页面
 * 提供策略配置、回测执行、结果分析、历史管理等完整功能
 */
const Backtest: React.FC = () => {
  // 状态管理
  const [activeTab, setActiveTab] = useState<string>('configure');
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [currentBacktest, setCurrentBacktest] =
    useState<BacktestResponse | null>(null);
  const [comparisonVisible, setComparisonVisible] = useState(false);
  const [selectedBacktests, setSelectedBacktests] = useState<string[]>([]);

  const { addNotification } = useAppStore();

  // 执行回测
  const handleRunBacktest = useCallback(
    async (request: BacktestRequest) => {
      setBacktestLoading(true);

      try {
        message.loading({ content: '正在提交回测请求...', key: 'backtest' });

        const response = await backtestApi.runBacktest(request);

        message.success({
          content: '回测请求提交成功',
          key: 'backtest',
          duration: 2,
        });

        setCurrentBacktest(response);
        setActiveTab('results');

        // 添加系统通知
        addNotification({
          type: 'success',
          title: '回测启动',
          message: `回测 "${request.backtest_name}" 已开始执行`,
        });

        // 如果是异步执行，显示进度提示
        if (request.async_execution) {
          Modal.info({
            title: '异步回测已启动',
            content: (
              <div>
                <p>回测 "{request.backtest_name}" 正在后台执行中...</p>
                <p>您可以在结果页面查看实时进度，或继续配置其他回测。</p>
                <p>
                  <Text code>回测ID: {response.backtest_id}</Text>
                </p>
              </div>
            ),
          });
        }
      } catch (error: any) {
        console.error('回测执行失败:', error);

        const errorMessage = error?.detail || error?.message || '回测执行失败';
        message.error({ content: errorMessage, key: 'backtest' });

        addNotification({
          type: 'error',
          title: '回测失败',
          message: errorMessage,
        });
      } finally {
        setBacktestLoading(false);
      }
    },
    [addNotification]
  );

  // 创建样本回测
  const handleCreateSample = useCallback(async () => {
    setBacktestLoading(true);

    try {
      message.loading({ content: '正在创建样本回测...', key: 'sample' });

      const response = await backtestApi.createSampleBacktest();

      message.success({
        content: '样本回测创建成功',
        key: 'sample',
        duration: 2,
      });

      setCurrentBacktest(response);
      setActiveTab('results');

      addNotification({
        type: 'success',
        title: '样本回测完成',
        message: `样本回测已完成，收益率: ${(response.total_return * 100).toFixed(2)}%`,
      });
    } catch (error: any) {
      console.error('创建样本回测失败:', error);

      const errorMessage =
        error?.detail || error?.message || '创建样本回测失败';
      message.error({ content: errorMessage, key: 'sample' });

      addNotification({
        type: 'error',
        title: '样本回测失败',
        message: errorMessage,
      });
    } finally {
      setBacktestLoading(false);
    }
  }, [addNotification]);

  // 处理回测对比
  const handleCompare = useCallback((backtestIds: string[]) => {
    setSelectedBacktests(backtestIds);
    setComparisonVisible(true);
  }, []);

  // 生成回测报告
  const handleGenerateReport = useCallback(
    async (backtestId: string) => {
      try {
        message.loading({ content: '正在生成报告...', key: 'report' });

        const reportResponse = await backtestApi.generateBacktestReport(
          backtestId,
          {
            report_format: 'html',
            include_charts: true,
            include_trades: true,
          }
        );

        message.success({
          content: '报告生成成功',
          key: 'report',
          duration: 2,
        });

        // 这里可以处理报告下载或显示
        console.log('报告生成结果:', reportResponse);

        addNotification({
          type: 'success',
          title: '报告生成完成',
          message: `回测 ${backtestId} 的分析报告已生成`,
        });
      } catch (error: any) {
        console.error('生成报告失败:', error);

        const errorMessage = error?.detail || error?.message || '生成报告失败';
        message.error({ content: errorMessage, key: 'report' });

        addNotification({
          type: 'error',
          title: '报告生成失败',
          message: errorMessage,
        });
      }
    },
    [addNotification]
  );

  // 渲染对比Modal
  const renderComparisonModal = () => {
    return (
      <Modal
        title="回测对比分析"
        open={comparisonVisible}
        onCancel={() => setComparisonVisible(false)}
        width={1200}
        footer={[
          <Button key="close" onClick={() => setComparisonVisible(false)}>
            关闭
          </Button>,
        ]}
      >
        <Alert
          message="回测对比功能"
          description={
            <div>
              <p>选中的回测ID: {selectedBacktests.join(', ')}</p>
              <p>回测对比功能正在开发中，将提供以下功能：</p>
              <ul>
                <li>收益曲线对比</li>
                <li>风险指标对比</li>
                <li>最大回撤对比</li>
                <li>夏普比率对比</li>
                <li>策略参数对比</li>
              </ul>
            </div>
          }
          type="info"
          showIcon
        />
      </Modal>
    );
  };

  return (
    <div style={{ padding: '24px', minHeight: '100vh' }}>
      {/* 页面头部 */}
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, marginBottom: '8px' }}>
          <Space>
            <ExperimentOutlined />
            回测工作台
          </Space>
        </Title>
        <Text type="secondary">
          专业的量化策略回测平台，支持多种策略类型和A股市场约束
        </Text>
      </div>

      {/* 主要内容区域 */}
      <Card size="small">
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            {
              key: 'configure',
              label: (
                <Space>
                  <PlayCircleOutlined />
                  策略配置
                </Space>
              ),
              children: (
                <div>
                  {/* 快速操作区 */}
                  <Card
                    title="快速开始"
                    size="small"
                    style={{ marginBottom: '16px' }}
                  >
                    <Space>
                      <Button
                        type="primary"
                        icon={<ExperimentOutlined />}
                        loading={backtestLoading}
                        onClick={handleCreateSample}
                      >
                        创建样本回测
                      </Button>

                      <Button
                        icon={<HistoryOutlined />}
                        onClick={() => setActiveTab('history')}
                      >
                        查看历史记录
                      </Button>

                      <Button
                        icon={<FileTextOutlined />}
                        onClick={() => {
                          Modal.info({
                            title: '使用指南',
                            width: 600,
                            content: (
                              <div>
                                <h4>回测工作台使用说明：</h4>
                                <ol>
                                  <li>
                                    <strong>选择策略：</strong>
                                    从支持的策略类型中选择合适的量化策略
                                  </li>
                                  <li>
                                    <strong>配置参数：</strong>
                                    根据策略类型设置相应的参数值
                                  </li>
                                  <li>
                                    <strong>选择股票池：</strong>
                                    可使用预设股票池或自定义股票列表
                                  </li>
                                  <li>
                                    <strong>设置时间范围：</strong>
                                    选择回测的起止日期
                                  </li>
                                  <li>
                                    <strong>交易配置：</strong>
                                    设置初始资金、手续费等交易参数
                                  </li>
                                  <li>
                                    <strong>A股约束：</strong>
                                    可选择启用涨跌停、T+1等A股市场约束
                                  </li>
                                  <li>
                                    <strong>执行回测：</strong>
                                    提交配置并等待回测完成
                                  </li>
                                  <li>
                                    <strong>分析结果：</strong>
                                    查看收益曲线、风险指标等结果
                                  </li>
                                </ol>
                                <Divider />
                                <h4>支持的策略类型：</h4>
                                <ul>
                                  <li>
                                    <strong>移动平均线穿越：</strong>
                                    基于快慢均线金叉死叉的趋势策略
                                  </li>
                                  <li>
                                    <strong>RSI均值回归：</strong>
                                    基于RSI超买超卖的反转策略
                                  </li>
                                  <li>
                                    <strong>动量策略：</strong>
                                    基于价格动量的趋势跟随策略
                                  </li>
                                </ul>
                              </div>
                            ),
                          });
                        }}
                      >
                        使用指南
                      </Button>
                    </Space>
                  </Card>

                  {/* 配置表单 */}
                  <BacktestForm
                    onSubmit={handleRunBacktest}
                    loading={backtestLoading}
                  />
                </div>
              ),
            },
            {
              key: 'results',
              label: (
                <Space>
                  <FileTextOutlined />
                  回测结果
                  {currentBacktest && (
                    <span
                      style={{
                        fontSize: '12px',
                        backgroundColor: '#f0f2ff',
                        color: '#1890ff',
                        padding: '2px 6px',
                        borderRadius: '10px',
                      }}
                    >
                      当前
                    </span>
                  )}
                </Space>
              ),
              children: (
                <div>
                  {currentBacktest ? (
                    <BacktestResults
                      backtest={currentBacktest}
                      onCompare={backtestId => handleCompare([backtestId])}
                      onGenerateReport={handleGenerateReport}
                    />
                  ) : (
                    <Card>
                      <div
                        style={{
                          textAlign: 'center',
                          padding: '40px 0',
                          color: '#999',
                        }}
                      >
                        <ExperimentOutlined
                          style={{ fontSize: '48px', marginBottom: '16px' }}
                        />
                        <div style={{ fontSize: '16px', marginBottom: '8px' }}>
                          暂无回测结果
                        </div>
                        <div style={{ fontSize: '14px' }}>
                          请先在策略配置页面创建一个回测任务
                        </div>
                        <Button
                          type="primary"
                          style={{ marginTop: '16px' }}
                          onClick={() => setActiveTab('configure')}
                        >
                          开始配置
                        </Button>
                      </div>
                    </Card>
                  )}
                </div>
              ),
            },
            {
              key: 'history',
              label: (
                <Space>
                  <HistoryOutlined />
                  历史记录
                </Space>
              ),
              children: (
                <BacktestHistory
                  onCompare={handleCompare}
                  onGenerateReport={handleGenerateReport}
                />
              ),
            },
          ]}
        />
      </Card>

      {/* 对比Modal */}
      {renderComparisonModal()}

      {/* 系统状态提示 */}
      <Alert
        message="系统信息"
        description={
          <div>
            <ul style={{ marginBottom: 0, paddingLeft: '20px' }}>
              <li>回测引擎: Vectorbt 高性能回测框架</li>
              <li>市场约束: 支持A股涨跌停、T+1等真实市场约束</li>
              <li>交易成本: 内置多券商费率模型，精确计算交易成本</li>
              <li>策略库: 支持均线、RSI、动量等多种经典策略</li>
              <li>数据源: 基于AKShare的实时A股数据</li>
            </ul>
          </div>
        }
        type="info"
        showIcon
        style={{ marginTop: '24px' }}
      />
    </div>
  );
};

export default Backtest;
