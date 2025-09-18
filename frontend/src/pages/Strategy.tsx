import { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Alert,
  Statistic,
  Typography,
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SettingOutlined,
  LineChartOutlined,
  ExperimentOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import PageHeader from '@/components/PageHeader';
import { LineChart } from '@/components/Charts';
import type { TimeSeriesDataPoint } from '@/types/charts';
import type { Strategy } from '@/types/api';
import {
  strategyApi,
  websocketUtils,
  type StrategyStatusUpdateMessage,
} from '@/services';
import { useResponsive } from '@/hooks/useResponsive';

const { Text } = Typography;
const { Option } = Select;

// 策略数据类型定义已移至 @/types/api

/**
 * 策略管理页面
 * 提供策略的创建、编辑、运行、监控等完整功能
 */
const Strategy: React.FC = () => {
  const { isMobile, tableSize, tablePagination, chartHeight, mobileConfig } =
    useResponsive();
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(false);

  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(
    null
  );
  const [modalVisible, setModalVisible] = useState(false);
  const [wsStatus, setWsStatus] = useState<string>('disconnected');

  // 加载策略数据
  const loadStrategies = async () => {
    try {
      setLoading(true);
      const response = await strategyApi.getStrategies({
        page: 1,
        page_size: 50,
      });
      setStrategies(response.strategies || []);
    } catch (error) {
      console.error('加载策略数据失败:', error);
      // 可以在这里添加错误提示
    } finally {
      setLoading(false);
    }
  };

  // 处理策略状态更新的WebSocket消息
  const handleStrategyStatusUpdate = (message: StrategyStatusUpdateMessage) => {
    const { strategy_id, new_status, message: statusMessage } = message.data;

    // 更新本地策略列表中对应策略的状态
    setStrategies(prev =>
      prev.map(strategy =>
        strategy.id === strategy_id
          ? { ...strategy, status: new_status as any }
          : strategy
      )
    );

    // 显示状态更新通知
    Modal.success({
      title: '策略状态更新',
      content: statusMessage,
    });
  };

  // 组件初始化
  useEffect(() => {
    loadStrategies();

    // 连接WebSocket并订阅策略状态更新
    websocketUtils.connect();

    // 监听WebSocket连接状态变化
    websocketUtils.onStatusChange(status => {
      setWsStatus(status);
    });

    // 设置初始WebSocket状态
    setWsStatus(websocketUtils.getStatus());

    // 订阅策略状态更新消息
    const unsubscribe = websocketUtils.subscribe(
      'strategy_status_update',
      handleStrategyStatusUpdate
    );

    // 清理函数
    return () => {
      unsubscribe();
      // 注意：不要在这里断开WebSocket连接，其他页面可能还在使用
    };
  }, []);
  const [form] = Form.useForm();

  // 生成模拟收益曲线数据
  const generateReturnsCurve = (_strategy: Strategy): TimeSeriesDataPoint[] => {
    const data: TimeSeriesDataPoint[] = [];
    const startDate = new Date('2024-01-01');
    let currentValue = 1;

    for (let i = 0; i < 30; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);

      // 模拟收益波动
      const randomChange = (Math.random() - 0.5) * 0.02;
      currentValue *= 1 + randomChange;

      data.push({
        timestamp: date.toLocaleDateString('zh-CN'),
        value: (currentValue - 1) * 100, // 转换为百分比
      });
    }

    return data;
  };

  // 表格列配置
  const columns: ColumnsType<Strategy> = [
    {
      title: '策略名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Strategy) => (
        <Space direction="vertical" size={0}>
          <Text strong>{text}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => {
        const typeMap: Record<string, { label: string; color: string }> = {
          ma_crossover: { label: '均线穿越', color: 'blue' },
          rsi_mean_reversion: { label: 'RSI回归', color: 'green' },
          momentum: { label: '动量策略', color: 'orange' },
        };

        const config = typeMap[type] || { label: type, color: 'default' };
        return <Tag color={config.color}>{config.label}</Tag>;
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusMap: Record<string, { label: string; color: string }> = {
          active: { label: '运行中', color: 'success' },
          inactive: { label: '已停止', color: 'default' },
          testing: { label: '测试中', color: 'processing' },
        };

        const config = statusMap[status];
        return <Tag color={config.color}>{config.label}</Tag>;
      },
    },
    {
      title: '总收益率',
      dataIndex: 'totalReturn',
      key: 'totalReturn',
      render: (value: number) => (
        <Text style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          {(value * 100).toFixed(2)}%
        </Text>
      ),
      sorter: (a, b) => a.totalReturn - b.totalReturn,
    },
    {
      title: '年化收益',
      dataIndex: 'annualReturn',
      key: 'annualReturn',
      render: (value: number) => (
        <Text style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          {(value * 100).toFixed(2)}%
        </Text>
      ),
      sorter: (a, b) => a.annualReturn - b.annualReturn,
    },
    {
      title: '最大回撤',
      dataIndex: 'maxDrawdown',
      key: 'maxDrawdown',
      render: (value: number) => (
        <Text style={{ color: '#cf1322' }}>
          {(Math.abs(value) * 100).toFixed(2)}%
        </Text>
      ),
      sorter: (a, b) => Math.abs(a.maxDrawdown) - Math.abs(b.maxDrawdown),
    },
    {
      title: '夏普比率',
      dataIndex: 'sharpeRatio',
      key: 'sharpeRatio',
      render: (value: number) => value.toFixed(2),
      sorter: (a, b) => a.sharpeRatio - b.sharpeRatio,
    },
    {
      title: '操作',
      key: 'action',
      width: 200,
      render: (_, record: Strategy) => (
        <Space>
          <Button
            type="text"
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record)}
          >
            编辑
          </Button>

          <Button
            type="text"
            size="small"
            icon={
              record.status === 'active' ? (
                <PauseCircleOutlined />
              ) : (
                <PlayCircleOutlined />
              )
            }
            onClick={() => handleToggleStatus(record)}
          >
            {record.status === 'active' ? '暂停' : '启动'}
          </Button>

          <Button
            type="text"
            size="small"
            icon={<ExperimentOutlined />}
            onClick={() => handleBacktest(record)}
          >
            回测
          </Button>

          <Button
            type="text"
            size="small"
            icon={<DeleteOutlined />}
            danger
            onClick={() => handleDelete(record)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  // 处理编辑策略
  const handleEdit = (strategy: Strategy) => {
    setSelectedStrategy(strategy);
    form.setFieldsValue(strategy);
    setModalVisible(true);
  };

  // 处理策略启动
  const handleStart = async (strategy: Strategy) => {
    try {
      setLoading(true);
      const result = await strategyApi.startStrategy(strategy.id);
      if (result?.success) {
        // 刷新策略列表
        await loadStrategies();
        Modal.success({
          title: '操作成功',
          content: `策略 "${strategy.name}" 已成功启动`,
        });
      }
    } catch (error) {
      console.error('启动策略失败:', error);
      Modal.error({
        title: '启动失败',
        content: `启动策略 "${strategy.name}" 失败，请检查策略配置`,
      });
    } finally {
      setLoading(false);
    }
  };

  // 处理策略停止
  const handleStop = async (strategy: Strategy) => {
    try {
      setLoading(true);
      const result = await strategyApi.stopStrategy(strategy.id);
      if (result?.success) {
        // 刷新策略列表
        await loadStrategies();
        Modal.success({
          title: '操作成功',
          content: `策略 "${strategy.name}" 已成功停止`,
        });
      }
    } catch (error) {
      console.error('停止策略失败:', error);
      Modal.error({
        title: '停止失败',
        content: `停止策略 "${strategy.name}" 失败`,
      });
    } finally {
      setLoading(false);
    }
  };

  // 处理策略暂停
  const handlePause = async (strategy: Strategy) => {
    try {
      setLoading(true);
      const result = await strategyApi.pauseStrategy(strategy.id);
      if (result?.success) {
        // 刷新策略列表
        await loadStrategies();
        Modal.success({
          title: '操作成功',
          content: `策略 "${strategy.name}" 已成功暂停`,
        });
      }
    } catch (error) {
      console.error('暂停策略失败:', error);
      Modal.error({
        title: '暂停失败',
        content: `暂停策略 "${strategy.name}" 失败`,
      });
    } finally {
      setLoading(false);
    }
  };

  // 处理切换状态
  const handleToggleStatus = (strategy: Strategy) => {
    if (strategy.status === 'active') {
      handlePause(strategy);
    } else {
      handleStart(strategy);
    }
  };

  // 处理回测
  const handleBacktest = (strategy: Strategy) => {
    Modal.info({
      title: '启动回测',
      content: `正在为策略 "${strategy.name}" 启动回测...`,
      onOk: () => {
        // 这里可以跳转到回测页面或调用回测接口
        window.location.href = '/backtest';
      },
    });
  };

  // 处理删除
  const handleDelete = (strategy: Strategy) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除策略 "${strategy.name}" 吗？此操作不可恢复。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          setLoading(true);
          const result = await strategyApi.deleteStrategy(strategy.id);
          if (result?.success) {
            // 刷新策略列表
            await loadStrategies();
            Modal.success({
              title: '删除成功',
              content: `策略 "${strategy.name}" 已成功删除`,
            });
          }
        } catch (error) {
          console.error('删除策略失败:', error);
          Modal.error({
            title: '删除失败',
            content: `删除策略 "${strategy.name}" 失败`,
          });
        } finally {
          setLoading(false);
        }
      },
    });
  };

  // 处理创建/编辑表单提交
  const handleSubmit = async (values: any) => {
    try {
      setLoading(true);
      if (selectedStrategy) {
        // 编辑现有策略
        const result = await strategyApi.updateStrategy(
          selectedStrategy.id,
          values
        );
        if (result) {
          await loadStrategies();
          setModalVisible(false);
          setSelectedStrategy(null);
          form.resetFields();
          Modal.success({
            title: '更新成功',
            content: `策略 "${result.name}" 已成功更新`,
          });
        }
      } else {
        // 创建新策略
        const result = await strategyApi.createStrategy(values);
        if (result) {
          await loadStrategies();
          setModalVisible(false);
          form.resetFields();
          Modal.success({
            title: '创建成功',
            content: `策略 "${result.name}" 已成功创建`,
          });
        }
      }
    } catch (error) {
      console.error('策略操作失败:', error);
      Modal.error({
        title: selectedStrategy ? '更新失败' : '创建失败',
        content: `${selectedStrategy ? '更新' : '创建'}策略失败，请检查输入信息`,
      });
    } finally {
      setLoading(false);
    }
  };

  const activeStrategies = strategies.filter(s => s.status === 'active');
  const testingStrategies = strategies.filter(s => s.status === 'testing');
  const totalReturn =
    strategies.reduce((sum, s) => sum + s.totalReturn, 0) / strategies.length;

  return (
    <div
      style={{ padding: mobileConfig.padding }}
      className={
        isMobile ? 'mobile-optimized mobile-performance' : 'desktop-optimized'
      }
    >
      <PageHeader
        title="策略管理"
        subtitle="管理和监控量化交易策略的运行状态与绩效"
        extra={
          <Space>
            {/* WebSocket连接状态指示器 */}
            <Space>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor:
                    wsStatus === 'connected'
                      ? '#52c41a'
                      : wsStatus === 'connecting'
                        ? '#faad14'
                        : '#ff4d4f',
                  display: 'inline-block',
                }}
              />
              <Text style={{ fontSize: 12, color: '#666' }}>
                实时连接:{' '}
                {wsStatus === 'connected'
                  ? '已连接'
                  : wsStatus === 'connecting'
                    ? '连接中'
                    : '已断开'}
              </Text>
            </Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => {
                setSelectedStrategy(null);
                form.resetFields();
                setModalVisible(true);
              }}
            >
              创建策略
            </Button>
          </Space>
        }
      />

      {/* 统计概览 */}
      <Row
        gutter={isMobile ? [8, 8] : [16, 16]}
        style={{ marginBottom: isMobile ? '16px' : '24px' }}
      >
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="策略总数"
              value={strategies.length}
              suffix="个"
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="运行中"
              value={activeStrategies.length}
              suffix="个"
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="测试中"
              value={testingStrategies.length}
              suffix="个"
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="平均收益率"
              value={(totalReturn * 100).toFixed(2)}
              suffix="%"
              valueStyle={{ color: totalReturn >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 策略列表 */}
        <Col xs={24} lg={16}>
          <Card
            title="策略列表"
            extra={
              <Space>
                <Button icon={<SettingOutlined />} size="small">
                  设置
                </Button>
                <Button icon={<FileTextOutlined />} size="small">
                  导出
                </Button>
              </Space>
            }
          >
            <Table
              columns={columns}
              dataSource={strategies}
              rowKey="id"
              size={tableSize}
              scroll={isMobile ? { x: 800 } : undefined}
              pagination={{
                pageSize: tablePagination.pageSize,
                showSizeChanger: tablePagination.showSizeChanger,
                showQuickJumper: tablePagination.showQuickJumper,
                showTotal: tablePagination.showTotal
                  ? total => `共 ${total} 个策略`
                  : undefined,
                simple: isMobile,
              }}
            />
          </Card>
        </Col>

        {/* 性能监控 */}
        <Col xs={24} lg={8}>
          <Card
            title="策略性能监控"
            size="small"
            style={{ marginBottom: '16px' }}
          >
            {activeStrategies.length > 0 ? (
              <LineChart
                data={generateReturnsCurve(activeStrategies[0])}
                title="收益曲线"
                height={200}
                color="#1890ff"
                smooth
                area
                formatter={value => `${value.toFixed(2)}%`}
              />
            ) : (
              <div
                style={{
                  textAlign: 'center',
                  padding: '40px 0',
                  color: '#999',
                }}
              >
                <LineChartOutlined
                  style={{ fontSize: '32px', marginBottom: '16px' }}
                />
                <div>暂无运行中的策略</div>
                <div style={{ fontSize: '12px' }}>启动策略后可查看性能监控</div>
              </div>
            )}
          </Card>

          <Alert
            message="策略运行提示"
            description="策略运行状态会实时更新，建议定期检查策略绩效并及时调整参数。"
            type="info"
            showIcon
            closable
          />
        </Col>
      </Row>

      {/* 创建/编辑策略Modal */}
      <Modal
        title={selectedStrategy ? '编辑策略' : '创建策略'}
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          setSelectedStrategy(null);
          form.resetFields();
        }}
        onOk={() => form.submit()}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleSubmit}>
          <Form.Item
            label="策略名称"
            name="name"
            rules={[{ required: true, message: '请输入策略名称' }]}
          >
            <Input placeholder="输入策略名称" />
          </Form.Item>

          <Form.Item
            label="策略类型"
            name="type"
            rules={[{ required: true, message: '请选择策略类型' }]}
          >
            <Select placeholder="选择策略类型">
              <Option value="ma_crossover">移动平均线穿越</Option>
              <Option value="rsi_mean_reversion">RSI均值回归</Option>
              <Option value="momentum">动量策略</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="策略描述"
            name="description"
            rules={[{ required: true, message: '请输入策略描述' }]}
          >
            <Input.TextArea rows={3} placeholder="描述策略的核心逻辑和特点" />
          </Form.Item>

          <Form.Item label="自动启动" name="autoStart" valuePropName="checked">
            <Switch checkedChildren="是" unCheckedChildren="否" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Strategy;
