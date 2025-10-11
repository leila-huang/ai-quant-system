/**
 * 数据中心页面
 *
 * 提供独立的数据管理功能，包括：
 * - 交易日历查看
 * - 高级股票池筛选
 * - 实时行情监控
 * - 财务数据查询
 * - 行业分类浏览
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Tabs,
  Table,
  Calendar,
  Select,
  InputNumber,
  Button,
  Space,
  Tag,
  Row,
  Col,
  Statistic,
  message,
  Badge,
  Tooltip,
  Input,
} from 'antd';
import type { Dayjs } from 'dayjs';
import dayjs from 'dayjs';
import {
  CalendarOutlined,
  FilterOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  ReloadOutlined,
  SearchOutlined,
  DownloadOutlined,
} from '@ant-design/icons';
import { dataAPI } from '../services/api';

const { TabPane } = Tabs;
const { Option } = Select;
const { Search } = Input;

interface TradingDay {
  date: string;
  isTrading: boolean;
}

interface StockInfo {
  symbol: string;
  name: string;
  industry: string;
  latest_price: number;
  change_pct: number;
  market_value: number;
  pe_ratio: number;
  pb_ratio: number;
  roe: number;
}

interface FundamentalData {
  symbol: string;
  date: string;
  eps: number;
  roe: number;
  operating_revenue: number;
  net_profit: number;
  gross_profit_margin: number;
  debt_to_asset_ratio: number;
}

const DataCenter: React.FC = () => {
  // 交易日历
  const [tradingDays, setTradingDays] = useState<TradingDay[]>([]);
  const [selectedYear, setSelectedYear] = useState<number>(dayjs().year());

  // 股票筛选
  const [stockList, setStockList] = useState<StockInfo[]>([]);
  const [filters, setFilters] = useState({
    industry: undefined,
    minPrice: undefined,
    maxPrice: undefined,
    minMarketValue: undefined,
    maxMarketValue: undefined,
    minPE: undefined,
    maxPE: undefined,
    minROE: undefined,
  });
  const [loading, setLoading] = useState(false);

  // 财务数据
  const [fundamentalData, setFundamentalData] =
    useState<FundamentalData | null>(null);
  const [searchSymbol, setSearchSymbol] = useState('');

  // 加载交易日历
  const loadTradingCalendar = async (year: number) => {
    try {
      setLoading(true);
      const response = await dataAPI.getTradingCalendar(year);
      setTradingDays(response.data.trading_days || []);
      message.success(`加载${year}年交易日历成功`);
    } catch (error) {
      message.error('加载交易日历失败');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 筛选股票
  const filterStocks = async () => {
    try {
      setLoading(true);
      const response = await dataAPI.filterStocks(filters);
      setStockList(response.data.stocks || []);
      message.success(
        `找到${response.data.stocks?.length || 0}只符合条件的股票`
      );
    } catch (error) {
      message.error('筛选股票失败');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 查询财务数据
  const loadFundamentalData = async (symbol: string) => {
    if (!symbol) {
      message.warning('请输入股票代码');
      return;
    }

    try {
      setLoading(true);
      const response = await dataAPI.getFundamentalData(symbol);
      setFundamentalData(response.data);
      message.success('查询财务数据成功');
    } catch (error) {
      message.error('查询财务数据失败');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTradingCalendar(selectedYear);
  }, [selectedYear]);

  // 日历单元格渲染
  const dateCellRender = (value: Dayjs) => {
    const dateStr = value.format('YYYY-MM-DD');
    const isTrading = tradingDays.some(
      day => day.date === dateStr && day.isTrading
    );

    if (isTrading) {
      return (
        <Badge
          status="success"
          text={
            <span style={{ fontSize: '12px', color: '#52c41a' }}>交易日</span>
          }
        />
      );
    }
    return null;
  };

  // 股票列表表格列定义
  const stockColumns = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
      fixed: 'left' as const,
    },
    {
      title: '股票名称',
      dataIndex: 'name',
      key: 'name',
      width: 120,
    },
    {
      title: '行业',
      dataIndex: 'industry',
      key: 'industry',
      width: 120,
      render: (industry: string) => <Tag color="blue">{industry}</Tag>,
    },
    {
      title: '最新价',
      dataIndex: 'latest_price',
      key: 'latest_price',
      width: 100,
      align: 'right' as const,
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '涨跌幅',
      dataIndex: 'change_pct',
      key: 'change_pct',
      width: 100,
      align: 'right' as const,
      render: (pct: number) => (
        <span style={{ color: pct >= 0 ? '#f5222d' : '#52c41a' }}>
          {pct >= 0 ? '+' : ''}
          {pct.toFixed(2)}%
        </span>
      ),
    },
    {
      title: '市值(亿)',
      dataIndex: 'market_value',
      key: 'market_value',
      width: 120,
      align: 'right' as const,
      render: (value: number) => (value / 100000000).toFixed(2),
    },
    {
      title: '市盈率',
      dataIndex: 'pe_ratio',
      key: 'pe_ratio',
      width: 100,
      align: 'right' as const,
      render: (value: number) => value?.toFixed(2) || '-',
    },
    {
      title: '市净率',
      dataIndex: 'pb_ratio',
      key: 'pb_ratio',
      width: 100,
      align: 'right' as const,
      render: (value: number) => value?.toFixed(2) || '-',
    },
    {
      title: 'ROE',
      dataIndex: 'roe',
      key: 'roe',
      width: 100,
      align: 'right' as const,
      render: (value: number) => (value ? `${value.toFixed(2)}%` : '-'),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Card
        title={
          <Space>
            <DatabaseOutlined />
            <span>数据中心</span>
          </Space>
        }
        extra={
          <Button
            type="primary"
            icon={<ReloadOutlined />}
            loading={loading}
            onClick={() => {
              loadTradingCalendar(selectedYear);
              message.success('数据已刷新');
            }}
          >
            刷新数据
          </Button>
        }
      >
        <Tabs defaultActiveKey="1">
          {/* 交易日历 */}
          <TabPane
            tab={
              <span>
                <CalendarOutlined />
                交易日历
              </span>
            }
            key="1"
          >
            <Card
              size="small"
              extra={
                <Select
                  value={selectedYear}
                  onChange={setSelectedYear}
                  style={{ width: 120 }}
                >
                  {[2023, 2024, 2025, 2026].map(year => (
                    <Option key={year} value={year}>
                      {year}年
                    </Option>
                  ))}
                </Select>
              }
            >
              <Row gutter={16} style={{ marginBottom: 16 }}>
                <Col span={8}>
                  <Statistic
                    title="交易日总数"
                    value={tradingDays.filter(d => d.isTrading).length}
                    suffix="天"
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="已过交易日"
                    value={
                      tradingDays.filter(
                        d => d.isTrading && dayjs(d.date).isBefore(dayjs())
                      ).length
                    }
                    suffix="天"
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="剩余交易日"
                    value={
                      tradingDays.filter(
                        d => d.isTrading && dayjs(d.date).isAfter(dayjs())
                      ).length
                    }
                    suffix="天"
                  />
                </Col>
              </Row>
              <Calendar
                dateCellRender={dateCellRender}
                headerRender={({ value, onChange }) => {
                  const month = value.month();
                  const year = value.year();
                  return (
                    <div style={{ padding: 8 }}>
                      <Row gutter={8}>
                        <Col>
                          <Select
                            size="small"
                            value={year}
                            onChange={newYear => {
                              const now = value.clone().year(newYear);
                              onChange(now);
                            }}
                          >
                            {[2023, 2024, 2025, 2026].map(y => (
                              <Option key={y} value={y}>
                                {y}年
                              </Option>
                            ))}
                          </Select>
                        </Col>
                        <Col>
                          <Select
                            size="small"
                            value={month}
                            onChange={newMonth => {
                              const now = value.clone().month(newMonth);
                              onChange(now);
                            }}
                          >
                            {Array.from({ length: 12 }, (_, i) => i).map(m => (
                              <Option key={m} value={m}>
                                {m + 1}月
                              </Option>
                            ))}
                          </Select>
                        </Col>
                      </Row>
                    </div>
                  );
                }}
              />
            </Card>
          </TabPane>

          {/* 高级筛选 */}
          <TabPane
            tab={
              <span>
                <FilterOutlined />
                股票筛选
              </span>
            }
            key="2"
          >
            <Card size="small" title="筛选条件" style={{ marginBottom: 16 }}>
              <Row gutter={[16, 16]}>
                <Col span={6}>
                  <label>行业：</label>
                  <Select
                    placeholder="选择行业"
                    style={{ width: '100%' }}
                    allowClear
                    onChange={value =>
                      setFilters({ ...filters, industry: value })
                    }
                  >
                    <Option value="金融">金融</Option>
                    <Option value="科技">科技</Option>
                    <Option value="医药">医药</Option>
                    <Option value="消费">消费</Option>
                    <Option value="工业">工业</Option>
                  </Select>
                </Col>
                <Col span={6}>
                  <label>价格范围：</label>
                  <Space>
                    <InputNumber
                      placeholder="最低价"
                      min={0}
                      onChange={value =>
                        setFilters({ ...filters, minPrice: value })
                      }
                    />
                    ~
                    <InputNumber
                      placeholder="最高价"
                      min={0}
                      onChange={value =>
                        setFilters({ ...filters, maxPrice: value })
                      }
                    />
                  </Space>
                </Col>
                <Col span={6}>
                  <label>市值范围（亿）：</label>
                  <Space>
                    <InputNumber
                      placeholder="最小市值"
                      min={0}
                      onChange={value =>
                        setFilters({ ...filters, minMarketValue: value })
                      }
                    />
                    ~
                    <InputNumber
                      placeholder="最大市值"
                      min={0}
                      onChange={value =>
                        setFilters({ ...filters, maxMarketValue: value })
                      }
                    />
                  </Space>
                </Col>
                <Col span={6}>
                  <label>最小ROE（%）：</label>
                  <InputNumber
                    placeholder="最小ROE"
                    min={0}
                    max={100}
                    style={{ width: '100%' }}
                    onChange={value =>
                      setFilters({ ...filters, minROE: value })
                    }
                  />
                </Col>
              </Row>
              <Row style={{ marginTop: 16 }}>
                <Col span={24}>
                  <Space>
                    <Button
                      type="primary"
                      icon={<SearchOutlined />}
                      onClick={filterStocks}
                      loading={loading}
                    >
                      开始筛选
                    </Button>
                    <Button
                      onClick={() => {
                        setFilters({
                          industry: undefined,
                          minPrice: undefined,
                          maxPrice: undefined,
                          minMarketValue: undefined,
                          maxMarketValue: undefined,
                          minPE: undefined,
                          maxPE: undefined,
                          minROE: undefined,
                        });
                        setStockList([]);
                      }}
                    >
                      重置
                    </Button>
                    <Button icon={<DownloadOutlined />}>导出结果</Button>
                  </Space>
                </Col>
              </Row>
            </Card>

            <Card size="small" title={`筛选结果（${stockList.length}只）`}>
              <Table
                columns={stockColumns}
                dataSource={stockList}
                rowKey="symbol"
                loading={loading}
                scroll={{ x: 1200 }}
                pagination={{
                  pageSize: 20,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: total => `共 ${total} 只股票`,
                }}
              />
            </Card>
          </TabPane>

          {/* 财务数据 */}
          <TabPane
            tab={
              <span>
                <LineChartOutlined />
                财务数据
              </span>
            }
            key="3"
          >
            <Card size="small" style={{ marginBottom: 16 }}>
              <Search
                placeholder="输入股票代码，如：600519"
                enterButton="查询"
                size="large"
                value={searchSymbol}
                onChange={e => setSearchSymbol(e.target.value)}
                onSearch={loadFundamentalData}
                loading={loading}
              />
            </Card>

            {fundamentalData && (
              <Card title={`${fundamentalData.symbol} 财务数据`}>
                <Row gutter={[16, 16]}>
                  <Col span={6}>
                    <Statistic
                      title="每股收益(EPS)"
                      value={fundamentalData.eps}
                      precision={2}
                      prefix="¥"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="净资产收益率(ROE)"
                      value={fundamentalData.roe}
                      precision={2}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="营业收入"
                      value={fundamentalData.operating_revenue / 100000000}
                      precision={2}
                      suffix="亿"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="净利润"
                      value={fundamentalData.net_profit / 100000000}
                      precision={2}
                      suffix="亿"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="毛利率"
                      value={fundamentalData.gross_profit_margin}
                      precision={2}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="资产负债率"
                      value={fundamentalData.debt_to_asset_ratio}
                      precision={2}
                      suffix="%"
                    />
                  </Col>
                  <Col span={12}>
                    <Tooltip title="数据日期">
                      <Tag
                        color="blue"
                        style={{ fontSize: 14, padding: '4px 8px' }}
                      >
                        数据日期: {fundamentalData.date}
                      </Tag>
                    </Tooltip>
                  </Col>
                </Row>
              </Card>
            )}
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default DataCenter;
