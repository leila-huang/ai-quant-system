import { useState, useRef, useEffect } from 'react';
import {
  Card,
  Input,
  Button,
  Space,
  Avatar,
  Typography,
  Row,
  Col,
  Tag,
  Alert,
  Spin,
  Tabs,
  Modal,
  List,
  Empty,
} from 'antd';
import {
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  BulbOutlined,
  BarChartOutlined,
  LineChartOutlined,
  FileTextOutlined,
  HistoryOutlined,
  ClearOutlined,
} from '@ant-design/icons';
import PageHeader from '@/components/PageHeader';
import type { ChatMessage, PresetQuestionItem } from '@/types/api';
import { aiApi } from '@/services';

const { Text, Paragraph } = Typography;
const { TextArea } = Input;

// 预设问题类型（扩展后端类型以包含React组件）
interface PresetQuestion {
  category: string;
  icon: React.ReactNode;
  questions: PresetQuestionItem[];
}

/**
 * AI助手页面
 * 提供智能问答、策略建议、市场分析等AI辅助功能
 */
const AI: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'assistant',
      content:
        '你好！我是AI量化助手，可以帮助您进行市场分析、策略优化和交易决策。请问有什么可以为您服务的吗？',
      timestamp: new Date().toISOString(),
      suggestions: ['查看市场趋势', '分析策略表现', '优化交易参数', '风险评估'],
    },
  ]);

  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [presetQuestions, setPresetQuestions] = useState<PresetQuestion[]>([]);
  const [loadingPresets, setLoadingPresets] = useState(false);
  const [sessionId] = useState(() => aiApi.generateSessionId());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [historyModalVisible, setHistoryModalVisible] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyMeta, setHistoryMeta] = useState<
    | {
        totalCount: number;
        message?: string;
      }
    | null
  >(null);

  // 加载预设问题数据
  const loadPresetQuestions = async () => {
    try {
      setLoadingPresets(true);
      const response = await aiApi.getPresetQuestions();
      if (response?.categories) {
        // 转换后端数据格式为前端组件需要的格式
        const mappedQuestions: PresetQuestion[] = response.categories.map(
          category => ({
            category: category.category,
            icon: getIconByCategory(category.category),
            questions: category.questions,
          })
        );
        setPresetQuestions(mappedQuestions);
      }
    } catch (error) {
      console.error('加载预设问题失败:', error);
      // 使用fallback数据
      setPresetQuestions(getFallbackPresetQuestions());
    } finally {
      setLoadingPresets(false);
    }
  };

  // 根据分类获取对应图标
  const getIconByCategory = (category: string): React.ReactNode => {
    const iconMap: { [key: string]: React.ReactNode } = {
      市场分析: <BarChartOutlined />,
      策略优化: <LineChartOutlined />,
      投研报告: <FileTextOutlined />,
    };
    return iconMap[category] || <BulbOutlined />;
  };

  // 获取fallback预设问题数据
  const getFallbackPresetQuestions = (): PresetQuestion[] => [
    {
      category: '市场分析',
      icon: <BarChartOutlined />,
      questions: [
        {
          title: '当前市场趋势分析',
          description: '分析当前A股市场的整体趋势和热点板块',
          prompt:
            '请分析当前A股市场的整体趋势，包括主要指数走势、热点板块和市场情绪指标。',
        },
        {
          title: '板块轮动分析',
          description: '识别当前市场的板块轮动规律',
          prompt:
            '请分析最近一个月的板块轮动情况，哪些板块表现强势，哪些板块相对落后？',
        },
        {
          title: '技术指标解读',
          description: '解读关键技术指标的信号',
          prompt:
            '请帮我解读当前沪深300指数的技术指标信号，包括MACD、RSI、均线等。',
        },
      ],
    },
    {
      category: '策略优化',
      icon: <LineChartOutlined />,
      questions: [
        {
          title: '策略参数调优',
          description: '优化现有量化策略的参数设置',
          prompt:
            '我的移动平均线策略最近表现不佳，请帮我分析可能的原因并建议参数优化方案。',
        },
        {
          title: '多因子模型构建',
          description: '构建多因子选股模型',
          prompt:
            '请帮我设计一个适合A股市场的多因子选股模型，包括因子选择和权重分配建议。',
        },
        {
          title: '风险控制建议',
          description: '完善策略的风险控制机制',
          prompt:
            '请为我的量化策略设计一套完整的风险控制体系，包括止损、仓位管理等。',
        },
      ],
    },
    {
      category: '投研报告',
      icon: <FileTextOutlined />,
      questions: [
        {
          title: '个股深度分析',
          description: '生成个股的投资分析报告',
          prompt:
            '请帮我分析贵州茅台(600519)的投资价值，包括基本面、技术面和估值分析。',
        },
        {
          title: '行业对比研究',
          description: '对比分析不同行业的投资机会',
          prompt: '请对比分析新能源汽车和传统汽车行业的投资机会和风险。',
        },
        {
          title: '量化因子研究',
          description: '研究特定量化因子的有效性',
          prompt:
            '请分析动量因子在A股市场的有效性，包括不同时间周期和市场环境下的表现。',
        },
      ],
    },
  ];

  // 滚动到消息底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // 组件初始化
  useEffect(() => {
    loadPresetQuestions();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 发送消息
  const handleSendMessage = async (content?: string) => {
    const messageContent = content || inputValue.trim();
    if (!messageContent) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: messageContent,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      // 调用真实的AI API
      const chatRequest = {
        message: messageContent,
        session_id: sessionId,
        context_type: 'general',
      };

      const response = await aiApi.chatWithAI(chatRequest);

      if (response?.message) {
        setMessages(prev => [...prev, response.message]);
      } else {
        // Fallback 错误处理
        const fallbackResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: '抱歉，我暂时无法处理您的请求，请稍后再试。',
          timestamp: new Date().toISOString(),
          suggestions: ['稍后重试', '查看帮助文档', '联系技术支持'],
        };
        setMessages(prev => [...prev, fallbackResponse]);
      }
    } catch (error) {
      console.error('AI聊天失败:', error);
      // 错误处理 - 添加错误消息
      const errorResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content:
          '网络连接异常，请检查网络后重试。如果问题持续存在，请联系技术支持。',
        timestamp: new Date().toISOString(),
        suggestions: ['检查网络连接', '稍后重试', '联系技术支持'],
      };
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setLoading(false);
    }
  };

  // 处理预设问题点击
  const handlePresetQuestion = (prompt: string) => {
    handleSendMessage(prompt);
  };

  // 处理建议点击
  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(`请帮我${suggestion}`);
  };

  const loadChatHistory = async () => {
    setHistoryError(null);
    setHistoryLoading(true);
    try {
      const history = await aiApi.getChatHistory(sessionId, 100);
      if (history) {
        setChatHistory(history.messages || []);
        setHistoryMeta({
          totalCount: history.total_count,
          message: history.message,
        });
      } else {
        setChatHistory([]);
        setHistoryMeta(null);
      }
    } catch (error) {
      console.error('加载对话历史失败:', error);
      setChatHistory([]);
      setHistoryMeta(null);
      setHistoryError('无法加载对话历史，请稍后重试。');
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleOpenHistory = () => {
    setHistoryModalVisible(true);
    loadChatHistory();
  };

  const handleCloseHistory = () => {
    setHistoryModalVisible(false);
  };

  const handleApplyHistory = () => {
    if (chatHistory.length > 0) {
      setMessages(chatHistory);
      setHistoryModalVisible(false);
    }
  };

  // 清空对话
  const handleClearChat = () => {
    setMessages([
      {
        id: '1',
        type: 'assistant',
        content: '对话已清空，有什么新的问题可以继续问我哦！',
        timestamp: new Date().toISOString(),
        suggestions: [
          '查看市场趋势',
          '分析策略表现',
          '优化交易参数',
          '风险评估',
        ],
      },
    ]);
  };

  return (
    <div style={{ padding: '24px' }}>
      <PageHeader
        title="AI助手"
        subtitle="智能量化分析助手，提供专业的投资建议和策略优化方案"
        extra={
          <Space>
            <Button
              icon={<HistoryOutlined />}
              onClick={handleOpenHistory}
              loading={historyLoading && historyModalVisible}
            >
              对话历史
            </Button>
            <Button icon={<ClearOutlined />} onClick={handleClearChat}>
              清空对话
            </Button>
          </Space>
        }
      />

      <Row gutter={[16, 16]}>
        {/* 对话区域 */}
        <Col xs={24} lg={16}>
          <Card
            title={
              <Space>
                <RobotOutlined style={{ color: '#1890ff' }} />
                <span>AI对话</span>
              </Space>
            }
            style={{ height: '600px' }}
            bodyStyle={{
              padding: '16px',
              height: 'calc(100% - 57px)',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* 消息列表 */}
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                marginBottom: '16px',
                padding: '0 8px',
              }}
            >
              {messages.map(message => (
                <div key={message.id} style={{ marginBottom: '16px' }}>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent:
                        message.type === 'user' ? 'flex-end' : 'flex-start',
                    }}
                  >
                    <div
                      style={{
                        maxWidth: '80%',
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '8px',
                        flexDirection:
                          message.type === 'user' ? 'row-reverse' : 'row',
                      }}
                    >
                      <Avatar
                        icon={
                          message.type === 'user' ? (
                            <UserOutlined />
                          ) : (
                            <RobotOutlined />
                          )
                        }
                        style={{
                          backgroundColor:
                            message.type === 'user' ? '#87d068' : '#1890ff',
                        }}
                      />

                      <div
                        style={{
                          padding: '8px 12px',
                          borderRadius: '8px',
                          backgroundColor:
                            message.type === 'user' ? '#e6f7ff' : '#f6ffed',
                          border: `1px solid ${message.type === 'user' ? '#91d5ff' : '#b7eb8f'}`,
                        }}
                      >
                        <Paragraph
                          style={{ margin: 0, whiteSpace: 'pre-wrap' }}
                        >
                          {message.content}
                        </Paragraph>

                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </Text>

                        {/* 建议按钮 */}
                        {message.suggestions &&
                          message.suggestions.length > 0 && (
                            <div style={{ marginTop: '8px' }}>
                              <Space wrap>
                                {message.suggestions.map(
                                  (suggestion, index) => (
                                    <Tag
                                      key={index}
                                      color="blue"
                                      style={{ cursor: 'pointer' }}
                                      onClick={() =>
                                        handleSuggestionClick(suggestion)
                                      }
                                    >
                                      {suggestion}
                                    </Tag>
                                  )
                                )}
                              </Space>
                            </div>
                          )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {loading && (
                <div style={{ textAlign: 'center', padding: '16px' }}>
                  <Spin>
                    <div style={{ padding: '20px' }}>
                      <Text type="secondary">AI正在思考中...</Text>
                    </div>
                  </Spin>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* 输入区域 */}
            <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '16px' }}>
              <Space.Compact style={{ width: '100%' }}>
                <TextArea
                  value={inputValue}
                  onChange={e => setInputValue(e.target.value)}
                  placeholder="输入您的问题，例如：当前市场趋势如何？"
                  autoSize={{ minRows: 1, maxRows: 4 }}
                  onPressEnter={e => {
                    if (!e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                />
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  loading={loading}
                  onClick={() => handleSendMessage()}
                  disabled={!inputValue.trim()}
                >
                  发送
                </Button>
              </Space.Compact>

              <Text
                type="secondary"
                style={{ fontSize: '12px', marginTop: '8px', display: 'block' }}
              >
                按 Enter 发送，Shift + Enter 换行
              </Text>
            </div>
          </Card>
        </Col>

        {/* 预设问题和帮助 */}
        <Col xs={24} lg={8}>
          <Card title="常见问题" style={{ marginBottom: '16px' }}>
            <Spin spinning={loadingPresets}>
              <Tabs
                size="small"
                items={presetQuestions.map((category, index) => ({
                  key: index.toString(),
                  label: (
                    <Space>
                      {category.icon}
                      <span>{category.category}</span>
                    </Space>
                  ),
                  children: (
                    <Space direction="vertical" style={{ width: '100%' }}>
                      {category.questions.map((question, qIndex) => (
                        <Card
                          key={qIndex}
                          size="small"
                          hoverable
                          onClick={() => handlePresetQuestion(question.prompt)}
                          style={{ cursor: 'pointer' }}
                        >
                          <Space direction="vertical" size={4}>
                            <Text strong style={{ fontSize: '14px' }}>
                              {question.title}
                            </Text>
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              {question.description}
                            </Text>
                          </Space>
                        </Card>
                      ))}
                    </Space>
                  ),
                }))}
              />
            </Spin>
          </Card>

          <Alert
            message="AI使用提示"
            description={
              <div>
                <p>• AI助手可以帮您分析市场趋势和优化交易策略</p>
                <p>• 建议描述具体的问题以获得更精准的回答</p>
                <p>• 可以点击建议标签快速提问相关问题</p>
                <p>• AI回答仅供参考，投资需谨慎</p>
              </div>
            }
            type="info"
            icon={<BulbOutlined />}
            showIcon
          />
        </Col>
      </Row>

      <Modal
        title="对话历史"
        open={historyModalVisible}
        onCancel={handleCloseHistory}
        width={640}
        footer={[
          <Button key="close" onClick={handleCloseHistory}>
            关闭
          </Button>,
          <Button
            key="apply"
            type="primary"
            onClick={handleApplyHistory}
            disabled={chatHistory.length === 0}
          >
            导入到当前会话
          </Button>,
        ]}
      >
        {historyLoading ? (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <Spin tip="正在加载对话历史..." />
          </div>
        ) : (
          <Space direction="vertical" style={{ width: '100%' }} size={16}>
            {historyError && (
              <Alert type="error" message={historyError} showIcon />
            )}

            {historyMeta?.message && !historyError && (
              <Alert type="info" message={historyMeta.message} showIcon />
            )}

            {chatHistory.length > 0 ? (
              <List
                size="small"
                dataSource={chatHistory}
                renderItem={item => (
                  <List.Item key={item.id} style={{ alignItems: 'flex-start' }}>
                    <List.Item.Meta
                      avatar={
                        <Avatar
                          icon={
                            item.type === 'user' ? <UserOutlined /> : <RobotOutlined />
                          }
                          style={{
                            backgroundColor:
                              item.type === 'user' ? '#87d068' : '#1890ff',
                          }}
                        />
                      }
                      title={
                        <Space split={<span>·</span>}>
                          <Text strong>
                            {item.type === 'user' ? '用户' : 'AI助手'}
                          </Text>
                          <Text type="secondary">
                            {new Date(item.timestamp).toLocaleString('zh-CN')}
                          </Text>
                        </Space>
                      }
                      description={
                        <Paragraph style={{ marginBottom: 8, whiteSpace: 'pre-wrap' }}>
                          {item.content}
                        </Paragraph>
                      }
                    />
                    {item.suggestions && item.suggestions.length > 0 && (
                      <Space wrap>
                        {item.suggestions.map((suggestion, index) => (
                          <Tag color="blue" key={index}>
                            {suggestion}
                          </Tag>
                        ))}
                      </Space>
                    )}
                  </List.Item>
                )}
              />
            ) : (
              <Empty description="暂无历史对话" />
            )}
          </Space>
        )}
      </Modal>
    </div>
  );
};

export default AI;
