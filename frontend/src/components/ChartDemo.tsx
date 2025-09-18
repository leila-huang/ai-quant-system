import React, { useState, useMemo } from 'react';
import { Card, Row, Col, Select, Button, Space, Typography } from 'antd';
import { LineChart, CandlestickChart, BarChart } from './Charts';
import type {
  TimeSeriesDataPoint,
  CandlestickDataPoint,
  BarDataPoint,
} from '@/types/charts';

const { Title, Paragraph } = Typography;

/**
 * 图表演示组件
 * 展示各种图表组件的功能和使用效果
 */
const ChartDemo: React.FC = () => {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [loading, setLoading] = useState(false);

  // 生成模拟的折线图数据
  const lineChartData = useMemo((): TimeSeriesDataPoint[] => {
    const data: TimeSeriesDataPoint[] = [];
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 30);

    for (let i = 0; i < 30; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);

      data.push({
        timestamp: date.toLocaleDateString('zh-CN'),
        value: 1000 + Math.random() * 500 + Math.sin(i / 5) * 200,
      });
    }
    return data;
  }, []);

  // 生成模拟的K线数据
  const candlestickData = useMemo((): CandlestickDataPoint[] => {
    const data: CandlestickDataPoint[] = [];
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 30);
    let basePrice = 100;

    for (let i = 0; i < 30; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);

      const change = (Math.random() - 0.5) * 10;
      const open = basePrice + change;
      const close = open + (Math.random() - 0.5) * 8;
      const high = Math.max(open, close) + Math.random() * 3;
      const low = Math.min(open, close) - Math.random() * 3;
      const volume = Math.random() * 1000000 + 500000;

      data.push({
        timestamp: date.toLocaleDateString('zh-CN'),
        open,
        close,
        high,
        low,
        volume,
      });

      basePrice = close;
    }
    return data;
  }, []);

  // 生成模拟的柱状图数据
  const barChartData = useMemo((): BarDataPoint[] => {
    const sectors = ['科技', '金融', '医药', '消费', '能源', '制造'];
    return sectors.map((sector, index) => ({
      name: sector,
      value: Math.random() * 100 + 50,
      color: `hsl(${index * 60}, 70%, 50%)`,
    }));
  }, []);

  const handleRefreshData = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  };

  return (
    <div style={{ padding: '16px' }}>
      <Card>
        <div style={{ marginBottom: '24px' }}>
          <Title level={3}>ECharts 图表组件演示</Title>
          <Paragraph type="secondary">
            展示专业的金融数据可视化图表组件，包括折线图、K线图、柱状图等。
          </Paragraph>

          <Space style={{ marginBottom: '16px' }}>
            <Select
              value={theme}
              onChange={setTheme}
              style={{ width: 120 }}
              options={[
                { label: '浅色主题', value: 'light' },
                { label: '深色主题', value: 'dark' },
              ]}
            />
            <Button onClick={handleRefreshData} loading={loading}>
              刷新数据
            </Button>
          </Space>
        </div>

        <Row gutter={[16, 16]}>
          {/* 折线图演示 */}
          <Col xs={24} lg={12}>
            <Card title="折线图 - 收益曲线" size="small">
              <LineChart
                data={lineChartData}
                title="投资组合收益"
                xAxisLabel="日期"
                yAxisLabel="收益金额(元)"
                height={300}
                smooth
                area
                theme={theme}
                loading={loading}
                formatter={value => `¥${value.toFixed(2)}`}
              />
            </Card>
          </Col>

          {/* 柱状图演示 */}
          <Col xs={24} lg={12}>
            <Card title="柱状图 - 行业分布" size="small">
              <BarChart
                data={barChartData}
                title="投资组合行业分布"
                xAxisLabel="行业"
                yAxisLabel="权重(%)"
                height={300}
                theme={theme}
                loading={loading}
                color={[
                  '#3b82f6',
                  '#ef4444',
                  '#22c55e',
                  '#f59e0b',
                  '#8b5cf6',
                  '#06b6d4',
                ]}
                formatter={value => `${value.toFixed(1)}%`}
              />
            </Card>
          </Col>

          {/* K线图演示 */}
          <Col xs={24}>
            <Card title="K线图 - 股票走势" size="small">
              <CandlestickChart
                data={candlestickData}
                title="股票价格走势"
                height={400}
                showVolume
                showMA
                maPeriods={[5, 10, 20]}
                theme={theme}
                loading={loading}
                formatter={{
                  price: value => `¥${value.toFixed(2)}`,
                  volume: value => {
                    if (value >= 10000) {
                      return `${(value / 10000).toFixed(1)}万`;
                    }
                    return value.toString();
                  },
                }}
              />
            </Card>
          </Col>
        </Row>

        {/* 功能说明 */}
        <Card title="图表功能特点" style={{ marginTop: '16px' }} size="small">
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Title level={5}>📈 折线图特性</Title>
              <ul style={{ fontSize: '14px', color: '#666' }}>
                <li>支持时间序列数据展示</li>
                <li>平滑曲线和面积填充</li>
                <li>数据缩放和交互</li>
                <li>自定义颜色和格式化</li>
              </ul>
            </Col>
            <Col xs={24} sm={8}>
              <Title level={5}>🕯️ K线图特性</Title>
              <ul style={{ fontSize: '14px', color: '#666' }}>
                <li>专业的股票K线展示</li>
                <li>成交量子图显示</li>
                <li>移动平均线指标</li>
                <li>丰富的交互功能</li>
              </ul>
            </Col>
            <Col xs={24} sm={8}>
              <Title level={5}>📊 柱状图特性</Title>
              <ul style={{ fontSize: '14px', color: '#666' }}>
                <li>支持垂直和水平布局</li>
                <li>多色彩分类显示</li>
                <li>数值标签显示</li>
                <li>动画效果优化</li>
              </ul>
            </Col>
          </Row>
        </Card>
      </Card>
    </div>
  );
};

export default ChartDemo;

