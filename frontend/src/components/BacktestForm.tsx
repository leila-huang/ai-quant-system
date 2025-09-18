import { useState, useEffect } from 'react';
import {
  Form,
  Input,
  Select,
  DatePicker,
  InputNumber,
  Switch,
  Card,
  Row,
  Col,
  Button,
  Space,
  Typography,
  Alert,
  Tooltip,
} from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import type { BacktestRequest, SupportedStrategiesResponse } from '@/types/api';
import { backtestApi } from '@/services';

const { Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
// TextArea unused, removed

interface BacktestFormProps {
  onSubmit: (values: BacktestRequest) => void;
  loading?: boolean;
  initialValues?: Partial<BacktestRequest>;
  className?: string;
}

/**
 * 回测配置表单组件
 * 提供完整的策略配置、参数设置、交易设置等功能
 */
const BacktestForm: React.FC<BacktestFormProps> = ({
  onSubmit,
  loading = false,
  initialValues,
  className,
}) => {
  const [form] = Form.useForm();
  const [supportedStrategies, setSupportedStrategies] =
    useState<SupportedStrategiesResponse | null>(null);
  const [selectedStrategy, setSelectedStrategy] =
    useState<string>('ma_crossover');
  const [customSymbols, setCustomSymbols] = useState<string[]>([]);
  const [strategiesLoading, setStrategiesLoading] = useState(false);

  // 预设的股票池
  const presetUniverses = [
    {
      label: '沪深300',
      value: 'hs300',
      symbols: [
        '000001.SZ',
        '000002.SZ',
        '600000.SH',
        '600036.SH',
        '600519.SH',
      ],
      description: '大盘蓝筹股',
    },
    {
      label: '中证500',
      value: 'zz500',
      symbols: [
        '000858.SZ',
        '002415.SZ',
        '300059.SZ',
        '300144.SZ',
        '300750.SZ',
      ],
      description: '中盘成长股',
    },
    {
      label: '科技股',
      value: 'tech',
      symbols: [
        '300059.SZ',
        '300750.SZ',
        '002415.SZ',
        '300144.SZ',
        '688111.SH',
      ],
      description: '科技行业股票',
    },
    {
      label: '消费股',
      value: 'consumer',
      symbols: [
        '600519.SH',
        '000858.SZ',
        '002304.SZ',
        '600887.SH',
        '000596.SZ',
      ],
      description: '消费行业股票',
    },
  ];

  // 获取支持的策略列表
  useEffect(() => {
    const fetchStrategies = async () => {
      setStrategiesLoading(true);
      try {
        const strategies = await backtestApi.getSupportedStrategies();
        setSupportedStrategies(strategies);
      } catch (error) {
        console.error('获取策略列表失败:', error);
      } finally {
        setStrategiesLoading(false);
      }
    };

    fetchStrategies();
  }, []);

  // 设置表单默认值
  useEffect(() => {
    const defaultValues: Partial<BacktestRequest> = {
      backtest_name: `回测_${dayjs().format('YYYYMMDD_HHmmss')}`,
      strategy_config: {
        strategy_type: 'ma_crossover',
        strategy_name: '移动平均线穿越策略',
        parameters: {
          fast_period: 5,
          slow_period: 20,
          stop_loss: 0.05,
        },
      },
      universe: presetUniverses[0].symbols,
      initial_capital: 1000000,
      commission: 0.0003,
      enable_cost_model: true,
      broker_type: 'standard',
      enable_constraints: true,
      price_limit_enabled: true,
      t_plus_1_enabled: true,
      rebalance_frequency: 'daily',
      benchmark_symbol: '000300.SH',
      async_execution: false,
      ...initialValues,
    };

    form.setFieldsValue({
      ...defaultValues,
      date_range: [dayjs().subtract(1, 'year'), dayjs().subtract(1, 'month')],
    });
  }, [form, initialValues]);

  // 处理策略类型变化
  const handleStrategyChange = (strategyType: string) => {
    setSelectedStrategy(strategyType);

    if (supportedStrategies?.supported_strategies[strategyType]) {
      const strategy = supportedStrategies.supported_strategies[strategyType];
      const defaultParams: Record<string, any> = {};

      Object.entries(strategy.parameters).forEach(([key, param]) => {
        defaultParams[key] = param.default;
      });

      form.setFieldsValue({
        strategy_config: {
          strategy_type: strategyType,
          strategy_name: strategy.name,
          parameters: defaultParams,
        },
      });
    }
  };

  // 处理股票池变化
  const handleUniverseChange = (universeType: string) => {
    const universe = presetUniverses.find(u => u.value === universeType);
    if (universe) {
      form.setFieldsValue({ universe: universe.symbols });
    }
  };

  // 添加自定义股票
  const handleAddSymbol = (symbol: string) => {
    if (symbol && !customSymbols.includes(symbol)) {
      const newSymbols = [...customSymbols, symbol];
      setCustomSymbols(newSymbols);

      const currentUniverse = form.getFieldValue('universe') || [];
      form.setFieldsValue({
        universe: [...new Set([...currentUniverse, symbol])],
      });
    }
  };

  // 表单提交处理
  const handleSubmit = (values: any) => {
    const { date_range, strategy_config, ...otherValues } = values;

    const request: BacktestRequest = {
      ...otherValues,
      start_date: date_range[0].format('YYYY-MM-DD'),
      end_date: date_range[1].format('YYYY-MM-DD'),
      strategy_config: {
        ...strategy_config,
        parameters: strategy_config.parameters || {},
      },
    };

    onSubmit(request);
  };

  // 渲染策略参数表单
  const renderStrategyParameters = () => {
    if (!supportedStrategies?.supported_strategies[selectedStrategy]) {
      return null;
    }

    const strategy = supportedStrategies.supported_strategies[selectedStrategy];
    const parameters = strategy.parameters;

    return (
      <Card title="策略参数" size="small" style={{ marginTop: '16px' }}>
        <Row gutter={[16, 16]}>
          {Object.entries(parameters).map(([key, param]) => (
            <Col xs={24} sm={12} lg={8} key={key}>
              <Form.Item
                label={
                  <Space>
                    <span>{param.description}</span>
                    <Tooltip title={`参数类型: ${param.type}`}>
                      <InfoCircleOutlined style={{ color: '#999' }} />
                    </Tooltip>
                  </Space>
                }
                name={['strategy_config', 'parameters', key]}
                rules={[
                  { required: true, message: `请输入${param.description}` },
                ]}
              >
                {param.type === 'int' || param.type === 'float' ? (
                  <InputNumber
                    style={{ width: '100%' }}
                    min={param.min}
                    max={param.max}
                    step={param.type === 'float' ? 0.01 : 1}
                    precision={param.type === 'float' ? 3 : 0}
                  />
                ) : param.options ? (
                  <Select>
                    {param.options.map((option: any) => (
                      <Option key={option} value={option}>
                        {option}
                      </Option>
                    ))}
                  </Select>
                ) : (
                  <Input />
                )}
              </Form.Item>
            </Col>
          ))}
        </Row>
      </Card>
    );
  };

  return (
    <div className={className}>
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        scrollToFirstError
      >
        {/* 基本信息 */}
        <Card title="基本信息" size="small">
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12}>
              <Form.Item
                label="回测名称"
                name="backtest_name"
                rules={[{ required: true, message: '请输入回测名称' }]}
              >
                <Input placeholder="输入回测名称" />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item
                label="回测时间范围"
                name="date_range"
                rules={[{ required: true, message: '请选择回测时间范围' }]}
              >
                <RangePicker
                  style={{ width: '100%' }}
                  format="YYYY-MM-DD"
                  disabledDate={current =>
                    current && current > dayjs().subtract(1, 'day')
                  }
                />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* 策略配置 */}
        <Card title="策略配置" size="small" style={{ marginTop: '16px' }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12}>
              <Form.Item
                label="策略类型"
                name={['strategy_config', 'strategy_type']}
                rules={[{ required: true, message: '请选择策略类型' }]}
              >
                <Select
                  loading={strategiesLoading}
                  onChange={handleStrategyChange}
                  placeholder="选择策略类型"
                >
                  {supportedStrategies &&
                    Object.entries(
                      supportedStrategies.supported_strategies
                    ).map(([key, strategy]) => (
                      <Option key={key} value={key}>
                        <div>
                          <div style={{ fontWeight: 'bold' }}>
                            {strategy.name}
                          </div>
                          <div style={{ fontSize: '12px', color: '#999' }}>
                            {strategy.description}
                          </div>
                        </div>
                      </Option>
                    ))}
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item
                label="策略名称"
                name={['strategy_config', 'strategy_name']}
                rules={[{ required: true, message: '请输入策略名称' }]}
              >
                <Input placeholder="输入策略名称" />
              </Form.Item>
            </Col>
          </Row>

          {renderStrategyParameters()}
        </Card>

        {/* 股票池配置 */}
        <Card title="股票池配置" size="small" style={{ marginTop: '16px' }}>
          <Row gutter={[16, 16]}>
            <Col xs={24}>
              <Space wrap style={{ marginBottom: '16px' }}>
                <Text>快速选择:</Text>
                {presetUniverses.map(universe => (
                  <Button
                    key={universe.value}
                    size="small"
                    onClick={() => handleUniverseChange(universe.value)}
                  >
                    {universe.label}
                    <Tooltip title={universe.description}>
                      <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                    </Tooltip>
                  </Button>
                ))}
              </Space>
            </Col>
            <Col xs={24}>
              <Form.Item
                label={
                  <Space>
                    <span>股票代码列表</span>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      (支持沪深A股代码，如: 000001.SZ, 600000.SH)
                    </Text>
                  </Space>
                }
                name="universe"
                rules={[
                  { required: true, message: '请选择股票' },
                  { type: 'array', min: 1, message: '至少选择一只股票' },
                ]}
              >
                <Select
                  mode="tags"
                  style={{ width: '100%' }}
                  placeholder="输入或选择股票代码"
                  tokenSeparators={[',', ' ', '\n']}
                  onSelect={handleAddSymbol}
                  maxTagCount="responsive"
                  showSearch
                  optionFilterProp="children"
                >
                  {presetUniverses
                    .flatMap(u => u.symbols)
                    .map(symbol => (
                      <Option key={symbol} value={symbol}>
                        {symbol}
                      </Option>
                    ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* 交易配置 */}
        <Card title="交易配置" size="small" style={{ marginTop: '16px' }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Form.Item
                label="初始资金"
                name="initial_capital"
                rules={[
                  { required: true, message: '请输入初始资金' },
                  {
                    type: 'number',
                    min: 10000,
                    message: '初始资金不能少于1万元',
                  },
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  formatter={value =>
                    `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
                  }
                  parser={value => value!.replace(/¥\s?|(,*)/g, '')}
                  step={10000}
                />
              </Form.Item>
            </Col>
            <Col xs={24} sm={8}>
              <Form.Item
                label="手续费率"
                name="commission"
                rules={[
                  { required: true, message: '请输入手续费率' },
                  {
                    type: 'number',
                    min: 0,
                    max: 0.01,
                    message: '手续费率应在0-1%之间',
                  },
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  step={0.0001}
                  precision={4}
                  formatter={value => `${(Number(value) * 100).toFixed(2)}%`}
                  parser={value => Number(value!.replace('%', '')) / 100}
                />
              </Form.Item>
            </Col>
            <Col xs={24} sm={8}>
              <Form.Item
                label="券商类型"
                name="broker_type"
                rules={[{ required: true, message: '请选择券商类型' }]}
              >
                <Select>
                  <Option value="standard">标准券商</Option>
                  <Option value="low_commission">低佣券商</Option>
                  <Option value="minimum">极低佣券商</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Form.Item
                label="调仓频率"
                name="rebalance_frequency"
                rules={[{ required: true, message: '请选择调仓频率' }]}
              >
                <Select>
                  <Option value="daily">每日</Option>
                  <Option value="weekly">每周</Option>
                  <Option value="monthly">每月</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} sm={8}>
              <Form.Item label="基准指数" name="benchmark_symbol">
                <Select allowClear placeholder="选择基准指数">
                  <Option value="000300.SH">沪深300</Option>
                  <Option value="000905.SH">中证500</Option>
                  <Option value="000001.SH">上证指数</Option>
                  <Option value="399001.SZ">深证成指</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} sm={8}>
              <Form.Item
                label="执行模式"
                name="async_execution"
                valuePropName="checked"
              >
                <Switch checkedChildren="异步" unCheckedChildren="同步" />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* A股约束配置 */}
        <Card title="A股约束配置" size="small" style={{ marginTop: '16px' }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Form.Item
                label="启用A股约束"
                name="enable_constraints"
                valuePropName="checked"
                tooltip="开启后将模拟真实A股市场的交易约束"
              >
                <Switch />
              </Form.Item>
            </Col>
            <Col xs={24} sm={8}>
              <Form.Item
                label="涨跌停约束"
                name="price_limit_enabled"
                valuePropName="checked"
                tooltip="模拟A股10%涨跌停限制"
              >
                <Switch />
              </Form.Item>
            </Col>
            <Col xs={24} sm={8}>
              <Form.Item
                label="T+1约束"
                name="t_plus_1_enabled"
                valuePropName="checked"
                tooltip="模拟A股T+1交易制度"
              >
                <Switch />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12}>
              <Form.Item
                label="交易成本模型"
                name="enable_cost_model"
                valuePropName="checked"
                tooltip="启用详细的交易成本计算模型"
              >
                <Switch />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* 提交按钮 */}
        <div style={{ marginTop: '24px', textAlign: 'center' }}>
          <Space>
            <Button onClick={() => form.resetFields()}>重置</Button>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              size="large"
            >
              开始回测
            </Button>
          </Space>
        </div>
      </Form>

      {/* 配置提示 */}
      <Alert
        message="回测配置提示"
        description={
          <ul style={{ marginBottom: 0 }}>
            <li>建议回测时间范围不少于1年，以获得更准确的策略表现评估</li>
            <li>股票池大小建议在5-50只之间，过多会影响回测性能</li>
            <li>异步执行模式适合大规模回测，同步模式适合快速验证</li>
            <li>启用A股约束将更真实地模拟实盘交易环境</li>
          </ul>
        }
        type="info"
        showIcon
        style={{ marginTop: '16px' }}
      />
    </div>
  );
};

export default BacktestForm;
