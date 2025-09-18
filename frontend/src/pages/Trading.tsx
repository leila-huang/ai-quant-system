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
  InputNumber,
  Alert,
  Statistic,
  Typography,
  Progress,
  List,
  message,
} from 'antd';
import {
  ShoppingCartOutlined,
  MinusCircleOutlined,
  RiseOutlined,
  FallOutlined,
  EyeOutlined,
  SettingOutlined,
  AlertOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import PageHeader from '@/components/PageHeader';
import { CandlestickChart } from '@/components/Charts';
import type { CandlestickDataPoint } from '@/types/charts';

const { Text } = Typography;

// 持仓数据类型定义
interface Position {
  id: string;
  symbol: string;
  name: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
  marketValue: number;
  cost: number;
  pnl: number;
  pnlRatio: number;
  side: 'long' | 'short';
}

// 交易记录类型定义
interface TradeRecord {
  id: string;
  symbol: string;
  name: string;
  side: 'buy' | 'sell';
  shares: number;
  price: number;
  amount: number;
  fee: number;
  timestamp: string;
  status: 'filled' | 'pending' | 'cancelled';
}

// 账户信息类型定义
interface AccountInfo {
  totalAssets: number;
  availableCash: number;
  marketValue: number;
  todayPnl: number;
  totalPnl: number;
  totalPnlRatio: number;
}

/**
 * 纸上交易页面
 * 提供模拟交易环境，包括买卖操作、持仓管理、交易记录等功能
 */
const Trading: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([
    {
      id: '1',
      symbol: '000001.SZ',
      name: '平安银行',
      shares: 1000,
      avgPrice: 12.5,
      currentPrice: 13.25,
      marketValue: 13250,
      cost: 12500,
      pnl: 750,
      pnlRatio: 0.06,
      side: 'long',
    },
    {
      id: '2',
      symbol: '600519.SH',
      name: '贵州茅台',
      shares: 100,
      avgPrice: 1680.0,
      currentPrice: 1720.0,
      marketValue: 172000,
      cost: 168000,
      pnl: 4000,
      pnlRatio: 0.024,
      side: 'long',
    },
  ]);

  const [tradeRecords, setTradeRecords] = useState<TradeRecord[]>([
    {
      id: '1',
      symbol: '000001.SZ',
      name: '平安银行',
      side: 'buy',
      shares: 1000,
      price: 12.5,
      amount: 12500,
      fee: 6.25,
      timestamp: '2024-01-20 09:30:15',
      status: 'filled',
    },
    {
      id: '2',
      symbol: '600519.SH',
      name: '贵州茅台',
      side: 'buy',
      shares: 100,
      price: 1680.0,
      amount: 168000,
      fee: 84.0,
      timestamp: '2024-01-19 14:25:30',
      status: 'filled',
    },
  ]);

  const [accountInfo, setAccountInfo] = useState<AccountInfo>({
    totalAssets: 285250,
    availableCash: 100000,
    marketValue: 185250,
    todayPnl: 2500,
    totalPnl: 4750,
    totalPnlRatio: 0.0169,
  });

  const [tradeModalVisible, setTradeModalVisible] = useState(false);
  const [tradeType, setTradeType] = useState<'buy' | 'sell'>('buy');
  const [form] = Form.useForm();
  const [settingsForm] = Form.useForm();
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(
    null
  );
  const [tradingConfig, setTradingConfig] = useState({
    feeRate: 0.0005,
    minLot: 100,
  });

  // 模拟股价更新
  useEffect(() => {
    const interval = setInterval(() => {
      setPositions(prev =>
        prev.map(pos => {
          const priceChange = (Math.random() - 0.5) * 0.02;
          const newPrice = pos.currentPrice * (1 + priceChange);
          const newMarketValue = pos.shares * newPrice;
          const newPnl = newMarketValue - pos.cost;
          const newPnlRatio = newPnl / pos.cost;

          return {
            ...pos,
            currentPrice: newPrice,
            marketValue: newMarketValue,
            pnl: newPnl,
            pnlRatio: newPnlRatio,
          };
        })
      );
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // 更新账户信息
  useEffect(() => {
    const totalMarketValue = positions.reduce(
      (sum, pos) => sum + pos.marketValue,
      0
    );
    const totalPnl = positions.reduce((sum, pos) => sum + pos.pnl, 0);
    const totalCost = positions.reduce((sum, pos) => sum + pos.cost, 0);

    setAccountInfo(prev => ({
      ...prev,
      marketValue: totalMarketValue,
      totalAssets: prev.availableCash + totalMarketValue,
      totalPnl,
      totalPnlRatio: totalCost > 0 ? totalPnl / totalCost : 0,
    }));
  }, [positions]);

  // 生成模拟K线数据
  const generateCandlestickData = (): CandlestickDataPoint[] => {
    const data: CandlestickDataPoint[] = [];
    let basePrice = 13.0;

    for (let i = 0; i < 30; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (29 - i));

      const open = basePrice;
      const change = (Math.random() - 0.5) * 0.5;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * 0.2;
      const low = Math.min(open, close) - Math.random() * 0.2;
      const volume = Math.floor(Math.random() * 1000000) + 500000;

      data.push({
        timestamp: date.toLocaleDateString('zh-CN'),
        open,
        high,
        low,
        close,
        volume,
      });

      basePrice = close;
    }

    return data;
  };

  // 持仓表格列配置
  const positionColumns: ColumnsType<Position> = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (text: string, record: Position) => (
        <Space direction="vertical" size={0}>
          <Text strong>{text}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.name}
          </Text>
        </Space>
      ),
    },
    {
      title: '持仓量',
      dataIndex: 'shares',
      key: 'shares',
      render: (shares: number) => shares.toLocaleString(),
    },
    {
      title: '成本价',
      dataIndex: 'avgPrice',
      key: 'avgPrice',
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '现价',
      dataIndex: 'currentPrice',
      key: 'currentPrice',
      render: (price: number, record: Position) => {
        const priceChange = price - record.avgPrice;
        const changeRatio = priceChange / record.avgPrice;
        const color = changeRatio >= 0 ? '#3f8600' : '#cf1322';

        return (
          <Space direction="vertical" size={0}>
            <Text style={{ color }}>{`¥${price.toFixed(2)}`}</Text>
            <Text style={{ color, fontSize: '12px' }}>
              {changeRatio >= 0 ? '+' : ''}
              {(changeRatio * 100).toFixed(2)}%
            </Text>
          </Space>
        );
      },
    },
    {
      title: '市值',
      dataIndex: 'marketValue',
      key: 'marketValue',
      render: (value: number) => `¥${value.toLocaleString()}`,
    },
    {
      title: '盈亏',
      key: 'pnl',
      render: (_, record: Position) => {
        const color = record.pnl >= 0 ? '#3f8600' : '#cf1322';
        const icon = record.pnl >= 0 ? <RiseOutlined /> : <FallOutlined />;

        return (
          <Space direction="vertical" size={0}>
            <Text style={{ color }}>
              {icon} ¥{Math.abs(record.pnl).toLocaleString()}
            </Text>
            <Text style={{ color, fontSize: '12px' }}>
              {(record.pnlRatio * 100).toFixed(2)}%
            </Text>
          </Space>
        );
      },
    },
    {
      title: '操作',
      key: 'action',
      width: 150,
      render: (_, record: Position) => (
        <Space>
          <Button
            type="text"
            size="small"
            icon={<MinusCircleOutlined />}
            onClick={() => handleSell(record)}
          >
            卖出
          </Button>
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetail(record)}
          >
            详情
          </Button>
        </Space>
      ),
    },
  ];

  // 交易记录表格列配置
  const tradeColumns: ColumnsType<TradeRecord> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
    },
    {
      title: '股票',
      key: 'stock',
      render: (_, record: TradeRecord) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.symbol}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.name}
          </Text>
        </Space>
      ),
    },
    {
      title: '方向',
      dataIndex: 'side',
      key: 'side',
      render: (side: string) => (
        <Tag color={side === 'buy' ? 'red' : 'green'}>
          {side === 'buy' ? '买入' : '卖出'}
        </Tag>
      ),
    },
    {
      title: '数量',
      dataIndex: 'shares',
      key: 'shares',
      render: (shares: number) => shares.toLocaleString(),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '金额',
      dataIndex: 'amount',
      key: 'amount',
      render: (amount: number) => `¥${amount.toLocaleString()}`,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusMap: Record<string, { label: string; color: string }> = {
          filled: { label: '已成交', color: 'success' },
          pending: { label: '待成交', color: 'processing' },
          cancelled: { label: '已撤销', color: 'default' },
        };

        const config =
          statusMap[status] || ({ label: status, color: 'default' } as const);
        return <Tag color={config.color}>{config.label}</Tag>;
      },
    },
  ];

  // 处理买入
  const handleBuy = () => {
    setTradeType('buy');
    setSelectedPosition(null);
    form.resetFields();
    form.setFieldsValue({ shares: tradingConfig.minLot });
    setTradeModalVisible(true);
  };

  // 处理卖出
  const handleSell = (position: Position) => {
    setTradeType('sell');
    setSelectedPosition(position);
    form.setFieldsValue({
      symbol: position.symbol,
      shares: position.shares,
      price: Number(position.currentPrice.toFixed(2)),
    });
    setTradeModalVisible(true);
  };

  // 处理查看详情
  const handleViewDetail = (position: Position) => {
    Modal.info({
      title: `${position.name} (${position.symbol})`,
      width: 800,
      content: (
        <div>
          <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
            <Col span={12}>
              <CandlestickChart
                data={generateCandlestickData()}
                title="价格走势"
                height={300}
              />
            </Col>
            <Col span={12}>
              <Space
                direction="vertical"
                size="middle"
                style={{ width: '100%' }}
              >
                <Card size="small">
                  <Statistic
                    title="持仓数量"
                    value={position.shares}
                    suffix="股"
                  />
                </Card>
                <Card size="small">
                  <Statistic
                    title="成本价"
                    value={position.avgPrice}
                    prefix="¥"
                    precision={2}
                  />
                </Card>
                <Card size="small">
                  <Statistic
                    title="现价"
                    value={position.currentPrice}
                    prefix="¥"
                    precision={2}
                    valueStyle={{
                      color: position.pnl >= 0 ? '#3f8600' : '#cf1322',
                    }}
                  />
                </Card>
                <Card size="small">
                  <Statistic
                    title="盈亏"
                    value={position.pnl}
                    prefix="¥"
                    precision={0}
                    valueStyle={{
                      color: position.pnl >= 0 ? '#3f8600' : '#cf1322',
                    }}
                  />
                </Card>
              </Space>
            </Col>
          </Row>
        </div>
      ),
    });
  };

  const handleOpenSettings = () => {
    settingsForm.setFieldsValue({
      availableCash: accountInfo.availableCash,
      feeRate: tradingConfig.feeRate,
      minLot: tradingConfig.minLot,
    });
    setSettingsVisible(true);
  };

  const handleSettingsSubmit = (values: {
    availableCash: number;
    feeRate: number;
    minLot: number;
  }) => {
    setTradingConfig(prev => ({
      ...prev,
      feeRate: values.feeRate,
      minLot: values.minLot,
    }));

    setAccountInfo(prev => ({
      ...prev,
      availableCash: values.availableCash,
      totalAssets: values.availableCash + prev.marketValue,
    }));

    message.success('交易设置已更新');
    setSettingsVisible(false);
  };

  // 处理交易提交
  const handleTradeSubmit = (values: {
    symbol: string;
    shares: number;
    price: number;
  }) => {
    const symbol = values.symbol;
    const shares = Number(values.shares);
    const price = Number(values.price);

    if (!symbol || !shares || !price) {
      message.error('请输入完整的交易信息');
      return;
    }

    const amount = shares * price;
    const fee = amount * tradingConfig.feeRate;

    if (tradeType === 'buy') {
      if (shares % tradingConfig.minLot !== 0) {
        message.error(`交易数量需为 ${tradingConfig.minLot} 股的整数倍`);
        return;
      }

      const totalCost = amount + fee;
      if (totalCost > accountInfo.availableCash) {
        message.error('可用资金不足，无法完成买入');
        return;
      }

      setAccountInfo(prev => ({
        ...prev,
        availableCash: prev.availableCash - totalCost,
      }));

      const existingPosition = positions.find(p => p.symbol === symbol);
      if (existingPosition) {
        const totalShares = existingPosition.shares + shares;
        const newCost = existingPosition.cost + totalCost;
        const currentPrice = price;
        const marketValue = totalShares * currentPrice;
        const pnl = marketValue - newCost;

        setPositions(prev =>
          prev.map(p =>
            p.symbol === symbol
              ? {
                  ...p,
                  shares: totalShares,
                  avgPrice: newCost / totalShares,
                  cost: newCost,
                  currentPrice,
                  marketValue,
                  pnl,
                  pnlRatio: newCost > 0 ? pnl / newCost : 0,
                }
              : p
          )
        );
      } else {
        const totalCost = amount + fee;
        const newPosition: Position = {
          id: Date.now().toString(),
          symbol,
          name: symbol,
          shares,
          avgPrice: totalCost / shares,
          currentPrice: price,
          marketValue: amount,
          cost: totalCost,
          pnl: amount - totalCost,
          pnlRatio: totalCost > 0 ? (amount - totalCost) / totalCost : 0,
          side: 'long',
        };
        setPositions(prev => [...prev, newPosition]);
      }
    } else {
      const existingPosition = positions.find(p => p.symbol === symbol);
      if (!existingPosition) {
        message.error('当前未持有该股票，无法卖出');
        return;
      }

      if (shares > existingPosition.shares) {
        message.error('卖出数量不能超过持仓数量');
        return;
      }

      if (
        shares % tradingConfig.minLot !== 0 &&
        shares !== existingPosition.shares
      ) {
        message.error(`卖出数量需为 ${tradingConfig.minLot} 股的整数倍`);
        return;
      }

      const costPerShare = existingPosition.cost / existingPosition.shares;
      const realizedCost = costPerShare * shares;
      const realizedPnl = amount - fee - realizedCost;

      setAccountInfo(prev => ({
        ...prev,
        availableCash: prev.availableCash + (amount - fee),
        todayPnl: prev.todayPnl + realizedPnl,
      }));

      setPositions(prev =>
        prev
          .map(position => {
            if (position.symbol !== symbol) {
              return position;
            }

            const remainingShares = position.shares - shares;
            if (remainingShares <= 0) {
              return null;
            }

            const remainingCost = costPerShare * remainingShares;
            const currentPrice = price;
            const marketValue = remainingShares * currentPrice;
            const pnl = marketValue - remainingCost;

            return {
              ...position,
              shares: remainingShares,
              cost: remainingCost,
              currentPrice,
              marketValue,
              pnl,
              pnlRatio: remainingCost > 0 ? pnl / remainingCost : 0,
            };
          })
          .filter((position): position is Position => Boolean(position))
      );
    }

    const newTrade: TradeRecord = {
      id: Date.now().toString(),
      symbol,
      name: symbol,
      side: tradeType,
      shares,
      price,
      amount,
      fee,
      timestamp: new Date().toLocaleString('zh-CN'),
      status: 'filled',
    };

    setTradeRecords(prev => [newTrade, ...prev]);
    message.success('交易已提交');
    setTradeModalVisible(false);
    setSelectedPosition(null);
    form.resetFields();
  };

  return (
    <div style={{ padding: '24px' }}>
      <PageHeader
        title="纸上交易"
        subtitle="模拟交易环境，练习量化策略和交易技能，无真实资金风险"
        extra={
          <Space>
            <Button
              type="primary"
              icon={<ShoppingCartOutlined />}
              onClick={handleBuy}
            >
              买入
            </Button>
            <Button icon={<SettingOutlined />} onClick={handleOpenSettings}>
              交易设置
            </Button>
          </Space>
        }
      />

      {/* 账户概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="总资产"
              value={accountInfo.totalAssets}
              prefix="¥"
              precision={0}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="可用资金"
              value={accountInfo.availableCash}
              prefix="¥"
              precision={0}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="持仓市值"
              value={accountInfo.marketValue}
              prefix="¥"
              precision={0}
            />
          </Card>
        </Col>

        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="总盈亏"
              value={accountInfo.totalPnl}
              prefix="¥"
              precision={0}
              valueStyle={{
                color: accountInfo.totalPnl >= 0 ? '#3f8600' : '#cf1322',
              }}
              suffix={
                <Text
                  style={{
                    fontSize: '14px',
                    color: accountInfo.totalPnl >= 0 ? '#3f8600' : '#cf1322',
                  }}
                >
                  ({(accountInfo.totalPnlRatio * 100).toFixed(2)}%)
                </Text>
              }
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card
            title="持仓明细"
            extra={
              <Space>
                <Text type="secondary">共{positions.length}只股票</Text>
              </Space>
            }
            style={{ marginBottom: '16px' }}
          >
            <Table
              columns={positionColumns}
              dataSource={positions}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>

          <Card title="交易记录">
            <Table
              columns={tradeColumns}
              dataSource={tradeRecords}
              rowKey="id"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: total => `共 ${total} 条记录`,
              }}
              size="small"
            />
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="资产分布" style={{ marginBottom: '16px' }}>
            <div style={{ marginBottom: '16px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: '8px',
                }}
              >
                <Text>现金</Text>
                <Text>
                  {(
                    (accountInfo.availableCash / accountInfo.totalAssets) *
                    100
                  ).toFixed(1)}
                  %
                </Text>
              </div>
              <Progress
                percent={
                  (accountInfo.availableCash / accountInfo.totalAssets) * 100
                }
                strokeColor="#52c41a"
                showInfo={false}
              />
            </div>

            <div>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: '8px',
                }}
              >
                <Text>股票</Text>
                <Text>
                  {(
                    (accountInfo.marketValue / accountInfo.totalAssets) *
                    100
                  ).toFixed(1)}
                  %
                </Text>
              </div>
              <Progress
                percent={
                  (accountInfo.marketValue / accountInfo.totalAssets) * 100
                }
                strokeColor="#1890ff"
                showInfo={false}
              />
            </div>
          </Card>

          <Card title="风险提示" size="small">
            <List
              size="small"
              dataSource={[
                '纸上交易仅为模拟，实际交易存在风险',
                '建议设置止损点，控制单笔损失',
                '注意仓位管理，避免满仓操作',
                '定期评估策略效果并及时调整',
              ]}
              renderItem={item => (
                <List.Item>
                  <Space>
                    <AlertOutlined style={{ color: '#faad14' }} />
                    <Text style={{ fontSize: '12px' }}>{item}</Text>
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* 交易Modal */}
      <Modal
        title={tradeType === 'buy' ? '买入股票' : '卖出股票'}
        open={tradeModalVisible}
        onCancel={() => {
          setTradeModalVisible(false);
          form.resetFields();
          setSelectedPosition(null);
        }}
        onOk={() => form.submit()}
        width={500}
      >
        <Form form={form} layout="vertical" onFinish={handleTradeSubmit}>
          <Form.Item
            label="股票代码"
            name="symbol"
            rules={[{ required: true, message: '请输入股票代码' }]}
          >
            <Input
              placeholder="输入股票代码，如：000001.SZ"
              disabled={tradeType === 'sell'}
            />
          </Form.Item>

          <Form.Item
            label="交易数量"
            name="shares"
            rules={[
              { required: true, message: '请输入交易数量' },
              () => ({
                validator(_, value) {
                  if (!value || value <= 0) {
                    return Promise.reject(new Error('交易数量必须大于0'));
                  }
                  const isClosingPosition =
                    tradeType === 'sell' &&
                    selectedPosition &&
                    value === selectedPosition.shares;

                  if (
                    value % tradingConfig.minLot !== 0 &&
                    !isClosingPosition
                  ) {
                    return Promise.reject(
                      new Error(`交易数量需为 ${tradingConfig.minLot} 股的整数倍`)
                    );
                  }
                  if (
                    tradeType === 'sell' &&
                    selectedPosition &&
                    value > selectedPosition.shares
                  ) {
                    return Promise.reject(
                      new Error('卖出数量不能超过当前持仓数量')
                    );
                  }
                  return Promise.resolve();
                },
              }),
            ]}
          >
            <InputNumber
              placeholder="输入股数"
              min={tradingConfig.minLot}
              step={tradingConfig.minLot}
              max={
                tradeType === 'sell' && selectedPosition
                  ? selectedPosition.shares
                  : undefined
              }
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Form.Item
            label="交易价格"
            name="price"
            rules={[{ required: true, message: '请输入交易价格' }]}
          >
            <InputNumber
              placeholder="输入价格"
              min={0.01}
              step={0.01}
              precision={2}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Alert
            message="风险提示"
            description="纸上交易为模拟环境，实际交易请谨慎操作"
            type="warning"
            showIcon
            style={{ marginTop: '16px' }}
          />
        </Form>
      </Modal>

      <Modal
        title="交易设置"
        open={settingsVisible}
        onCancel={() => setSettingsVisible(false)}
        onOk={() => settingsForm.submit()}
        width={480}
      >
        <Form form={settingsForm} layout="vertical" onFinish={handleSettingsSubmit}>
          <Form.Item
            label="可用资金"
            name="availableCash"
            rules={[{ required: true, message: '请输入可用资金' }]}
          >
            <InputNumber
              min={0}
              step={1000}
              style={{ width: '100%' }}
              prefix="¥"
            />
          </Form.Item>

          <Form.Item
            label="单笔交易最小股数"
            name="minLot"
            rules={[{ required: true, message: '请输入最小交易股数' }]}
            extra="交易数量需为该数值的整数倍"
          >
            <InputNumber min={100} step={100} style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item
            label="交易手续费率"
            name="feeRate"
            rules={[{ required: true, message: '请输入手续费率' }]}
            extra="填写小数形式，例如0.0005表示0.05%"
          >
            <InputNumber
              min={0}
              max={0.01}
              step={0.0001}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Alert
            message="提示"
            description="交易设置将影响买卖校验、手续费和可用资金计算。"
            type="info"
            showIcon
          />
        </Form>
      </Modal>
    </div>
  );
};

export default Trading;
