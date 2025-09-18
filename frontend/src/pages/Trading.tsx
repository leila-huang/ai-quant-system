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

        const config = statusMap[status];
        return <Tag color={config.color}>{config.label}</Tag>;
      },
    },
  ];

  // 处理买入
  const handleBuy = () => {
    setTradeType('buy');
    setTradeModalVisible(true);
  };

  // 处理卖出
  const handleSell = (position: Position) => {
    setTradeType('sell');
    form.setFieldsValue({
      symbol: position.symbol,
      shares: position.shares,
      price: position.currentPrice,
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

  // 处理交易提交
  const handleTradeSubmit = (values: {
    symbol: string;
    shares: number;
    price: number;
  }) => {
    const newTrade: TradeRecord = {
      id: Date.now().toString(),
      symbol: values.symbol,
      name: values.symbol, // 简化处理
      side: tradeType,
      shares: values.shares,
      price: values.price,
      amount: values.shares * values.price,
      fee: values.shares * values.price * 0.0005, // 0.05% 手续费
      timestamp: new Date().toLocaleString('zh-CN'),
      status: 'filled',
    };

    setTradeRecords(prev => [newTrade, ...prev]);

    if (tradeType === 'buy') {
      // 买入逻辑
      const existingPosition = positions.find(p => p.symbol === values.symbol);
      if (existingPosition) {
        const totalShares = existingPosition.shares + values.shares;
        const totalCost = existingPosition.cost + newTrade.amount;
        const newAvgPrice = totalCost / totalShares;

        setPositions(prev =>
          prev.map(p =>
            p.symbol === values.symbol
              ? {
                  ...p,
                  shares: totalShares,
                  avgPrice: newAvgPrice,
                  cost: totalCost,
                  marketValue: totalShares * p.currentPrice,
                  pnl: totalShares * p.currentPrice - totalCost,
                  pnlRatio:
                    (totalShares * p.currentPrice - totalCost) / totalCost,
                }
              : p
          )
        );
      } else {
        const newPosition: Position = {
          id: Date.now().toString(),
          symbol: values.symbol,
          name: values.symbol,
          shares: values.shares,
          avgPrice: values.price,
          currentPrice: values.price,
          marketValue: newTrade.amount,
          cost: newTrade.amount,
          pnl: 0,
          pnlRatio: 0,
          side: 'long',
        };
        setPositions(prev => [...prev, newPosition]);
      }
    } else {
      // 卖出逻辑
      setPositions(prev =>
        prev
          .map(p =>
            p.symbol === values.symbol
              ? {
                  ...p,
                  shares: p.shares - values.shares,
                  marketValue: (p.shares - values.shares) * p.currentPrice,
                  // 保持成本价和成本不变，只减少持仓
                }
              : p
          )
          .filter(p => p.shares > 0)
      );
    }

    setTradeModalVisible(false);
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
            <Button icon={<SettingOutlined />}>交易设置</Button>
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
            rules={[{ required: true, message: '请输入交易数量' }]}
          >
            <InputNumber
              placeholder="输入股数"
              min={100}
              step={100}
              style={{ width: '100%' }}
              formatter={value => `${value}股`}
              parser={value =>
                (value ? Number(value.replace('股', '')) : 100) as 100
              }
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
              formatter={value => `¥ ${value}`}
              parser={value =>
                (value ? Number(value.replace('¥ ', '')) : 0.01) as 0.01
              }
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
    </div>
  );
};

export default Trading;
