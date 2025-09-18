import { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Input,
  Select,
  DatePicker,
  Modal,
  message,
  Tag,
  Tooltip,
  Popconfirm,
  Typography,
} from 'antd';
import {
  // SearchOutlined, - unused
  ReloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  SwapOutlined, // 替代CompareArrowsOutlined
  DownloadOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';
import type { BacktestMetadata, BacktestListParams } from '@/types/api';
import { backtestApi } from '@/services';
import BacktestResults from './BacktestResults';

const { RangePicker } = DatePicker;
const { Text } = Typography;

interface BacktestHistoryProps {
  onCompare?: (backtestIds: string[]) => void;
  onGenerateReport?: (backtestId: string) => void;
  className?: string;
}

/**
 * 回测历史管理组件
 * 提供回测记录的查询、筛选、对比、删除等功能
 */
const BacktestHistory: React.FC<BacktestHistoryProps> = ({
  onCompare,
  onGenerateReport,
  className,
}) => {
  const [backtests, setBacktests] = useState<BacktestMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const [detailVisible, setDetailVisible] = useState(false);
  const [selectedBacktest, setSelectedBacktest] =
    useState<BacktestMetadata | null>(null);

  // 筛选条件
  const [filters, setFilters] = useState<BacktestListParams>({
    limit: 20,
  });

  // 获取回测列表
  const fetchBacktests = useCallback(async () => {
    setLoading(true);
    try {
      const response = await backtestApi.getBacktestList(filters);
      setBacktests(response.backtests);
    } catch (error) {
      console.error('获取回测列表失败:', error);
      message.error('获取回测列表失败');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  // 组件挂载时获取数据
  useEffect(() => {
    fetchBacktests();
  }, [fetchBacktests]);

  // 删除回测
  const handleDelete = async (backtestId: string) => {
    try {
      await backtestApi.deleteBacktest(backtestId);
      message.success('删除成功');
      await fetchBacktests(); // 刷新列表
    } catch (error) {
      console.error('删除回测失败:', error);
      message.error('删除回测失败');
    }
  };

  // 批量删除
  const handleBatchDelete = async () => {
    if (selectedRowKeys.length === 0) {
      message.warning('请选择要删除的回测记录');
      return;
    }

    Modal.confirm({
      title: '确认删除',
      content: `确定要删除选中的 ${selectedRowKeys.length} 个回测记录吗？此操作不可恢复。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        for (const backtestId of selectedRowKeys) {
          try {
            await backtestApi.deleteBacktest(backtestId);
          } catch (error) {
            console.error(`删除回测 ${backtestId} 失败:`, error);
          }
        }
        message.success(`成功删除 ${selectedRowKeys.length} 个回测记录`);
        setSelectedRowKeys([]);
        await fetchBacktests();
      },
    });
  };

  // 批量对比
  const handleBatchCompare = () => {
    if (selectedRowKeys.length < 2) {
      message.warning('请至少选择2个回测记录进行对比');
      return;
    }

    if (selectedRowKeys.length > 10) {
      message.warning('最多只能同时对比10个回测记录');
      return;
    }

    if (onCompare) {
      onCompare(selectedRowKeys);
    }
  };

  // 查看详情
  const handleViewDetail = (backtest: BacktestMetadata) => {
    setSelectedBacktest(backtest);
    setDetailVisible(true);
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
    type: 'percent' | 'number' = 'number'
  ) => {
    if (typeof value !== 'number' || isNaN(value)) return '--';
    return type === 'percent'
      ? `${(value * 100).toFixed(2)}%`
      : value.toFixed(4);
  };

  // 表格列配置
  const columns: ColumnsType<BacktestMetadata> = [
    {
      title: '回测名称',
      dataIndex: 'backtest_name',
      key: 'backtest_name',
      width: 200,
      ellipsis: { showTitle: false },
      render: (text: string, record: BacktestMetadata) => (
        <Tooltip title={text}>
          <Button
            type="link"
            style={{ padding: 0, height: 'auto' }}
            onClick={() => handleViewDetail(record)}
          >
            {text}
          </Button>
        </Tooltip>
      ),
    },
    {
      title: '策略',
      dataIndex: ['strategy_config', 'strategy_name'],
      key: 'strategy_name',
      width: 150,
      ellipsis: true,
      render: (text: string, record: BacktestMetadata) => (
        <Tooltip title={record.strategy_config?.strategy_type}>{text}</Tooltip>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{getStatusText(status)}</Tag>
      ),
    },
    {
      title: '总收益率',
      dataIndex: 'total_return',
      key: 'total_return',
      width: 100,
      render: (value: number) => (
        <Text style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          {formatMetric(value, 'percent')}
        </Text>
      ),
      sorter: (a, b) => (a.total_return || 0) - (b.total_return || 0),
    },
    {
      title: '年化收益',
      dataIndex: 'annual_return',
      key: 'annual_return',
      width: 100,
      render: (value: number) => (
        <Text style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          {formatMetric(value, 'percent')}
        </Text>
      ),
      sorter: (a, b) => (a.annual_return || 0) - (b.annual_return || 0),
    },
    {
      title: '最大回撤',
      dataIndex: 'max_drawdown',
      key: 'max_drawdown',
      width: 100,
      render: (value: number) => (
        <Text style={{ color: '#cf1322' }}>
          {formatMetric(Math.abs(value || 0), 'percent')}
        </Text>
      ),
      sorter: (a, b) =>
        Math.abs(a.max_drawdown || 0) - Math.abs(b.max_drawdown || 0),
    },
    {
      title: '夏普比率',
      dataIndex: 'sharpe_ratio',
      key: 'sharpe_ratio',
      width: 100,
      render: (value: number) => formatMetric(value),
      sorter: (a, b) => (a.sharpe_ratio || 0) - (b.sharpe_ratio || 0),
    },
    {
      title: '交易笔数',
      dataIndex: 'trade_count',
      key: 'trade_count',
      width: 80,
      render: (value: number) => value || 0,
    },
    {
      title: '股票数',
      dataIndex: 'universe',
      key: 'universe',
      width: 80,
      render: (universe: string[]) => universe?.length || 0,
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      width: 120,
      render: (text: string) =>
        text ? dayjs(text).format('MM-DD HH:mm') : '--',
      sorter: (a, b) =>
        dayjs(a.started_at).valueOf() - dayjs(b.started_at).valueOf(),
    },
    {
      title: '操作',
      key: 'action',
      width: 150,
      fixed: 'right',
      render: (_, record: BacktestMetadata) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetail(record)}
            />
          </Tooltip>

          {record.status === 'completed' && onGenerateReport && (
            <Tooltip title="生成报告">
              <Button
                type="text"
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => onGenerateReport(record.backtest_id)}
              />
            </Tooltip>
          )}

          <Popconfirm
            title="确定要删除此回测记录吗？"
            onConfirm={() => handleDelete(record.backtest_id)}
            okText="删除"
            cancelText="取消"
          >
            <Tooltip title="删除">
              <Button
                type="text"
                size="small"
                icon={<DeleteOutlined />}
                danger
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className={className}>
      <Card
        title="回测历史"
        size="small"
        extra={
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchBacktests}
              loading={loading}
            >
              刷新
            </Button>

            {selectedRowKeys.length > 0 && (
              <>
                <Button
                  icon={<SwapOutlined />}
                  onClick={handleBatchCompare}
                  disabled={selectedRowKeys.length < 2}
                >
                  对比 ({selectedRowKeys.length})
                </Button>

                <Button
                  icon={<DeleteOutlined />}
                  onClick={handleBatchDelete}
                  danger
                >
                  批量删除
                </Button>
              </>
            )}
          </Space>
        }
      >
        {/* 筛选条件 */}
        <div style={{ marginBottom: '16px' }}>
          <Space wrap>
            <Input.Search
              placeholder="搜索回测名称或策略"
              allowClear
              style={{ width: 200 }}
              onSearch={value => {
                setFilters(prev => ({ ...prev, search: value || undefined }));
              }}
            />

            <Select
              placeholder="状态筛选"
              allowClear
              style={{ width: 120 }}
              onChange={value => {
                setFilters(prev => ({ ...prev, status_filter: value }));
              }}
            >
              <Select.Option value="completed">已完成</Select.Option>
              <Select.Option value="running">运行中</Select.Option>
              <Select.Option value="pending">等待中</Select.Option>
              <Select.Option value="failed">失败</Select.Option>
            </Select>

            <RangePicker
              placeholder={['开始日期', '结束日期']}
              onChange={dates => {
                if (dates && dates.length === 2) {
                  setFilters(prev => ({
                    ...prev,
                    start_date: dates[0]?.format('YYYY-MM-DD'),
                    end_date: dates[1]?.format('YYYY-MM-DD'),
                  }));
                } else {
                  setFilters(prev => {
                    const { start_date, end_date, ...rest } = prev;
                    return rest;
                  });
                }
              }}
            />
          </Space>
        </div>

        {/* 回测列表表格 */}
        <Table<BacktestMetadata>
          columns={columns}
          dataSource={backtests}
          rowKey="backtest_id"
          loading={loading}
          size="small"
          scroll={{ x: 1200 }}
          pagination={{
            total: backtests.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              `显示 ${range[0]}-${range[1]} 条，共 ${total} 条记录`,
          }}
          rowSelection={{
            selectedRowKeys,
            onChange: (keys: React.Key[]) =>
              setSelectedRowKeys(keys as string[]),
            selections: [
              Table.SELECTION_ALL,
              Table.SELECTION_INVERT,
              Table.SELECTION_NONE,
            ],
          }}
          summary={data => {
            if (data.length === 0) return null;

            const completedBacktests = data.filter(
              item => item.status === 'completed'
            );
            const avgReturn =
              completedBacktests.length > 0
                ? completedBacktests.reduce(
                    (sum, item) => sum + (item.total_return || 0),
                    0
                  ) / completedBacktests.length
                : 0;

            return (
              <Table.Summary.Row>
                <Table.Summary.Cell index={0}>
                  <Text strong>统计汇总</Text>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={1}>
                  <Text>已完成: {completedBacktests.length}</Text>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={2} />
                <Table.Summary.Cell index={3}>
                  <Text>平均收益: {formatMetric(avgReturn, 'percent')}</Text>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={4} />
                <Table.Summary.Cell index={5} />
                <Table.Summary.Cell index={6} />
                <Table.Summary.Cell index={7} />
                <Table.Summary.Cell index={8} />
                <Table.Summary.Cell index={9} />
                <Table.Summary.Cell index={10} />
              </Table.Summary.Row>
            );
          }}
        />
      </Card>

      {/* 详情Modal */}
      <Modal
        title="回测详情"
        open={detailVisible}
        onCancel={() => setDetailVisible(false)}
        width={1000}
        footer={[
          <Button key="close" onClick={() => setDetailVisible(false)}>
            关闭
          </Button>,
        ]}
      >
        {selectedBacktest && (
          <BacktestResults
            backtest={selectedBacktest}
            onCompare={backtestId => {
              setDetailVisible(false);
              if (onCompare) {
                onCompare([backtestId]);
              }
            }}
            onGenerateReport={onGenerateReport}
          />
        )}
      </Modal>
    </div>
  );
};

export default BacktestHistory;
