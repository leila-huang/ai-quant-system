import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Layout,
  Menu,
  Space,
  Typography,
  Button,
  Drawer,
  Breadcrumb,
} from 'antd';
import {
  DashboardOutlined,
  ExperimentOutlined,
  RobotOutlined,
  LineChartOutlined, // 替代TradingviewOutlined
  BarsOutlined,
  MenuOutlined,
  HomeOutlined,
  DatabaseOutlined,
} from '@ant-design/icons';
import WebSocketIndicator from './WebSocketIndicator';
import { useWebSocketConnection } from '@/stores/websocketStore';
import { useResponsive } from '@/hooks/useResponsive';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

interface MainLayoutProps {
  children: React.ReactNode;
}

/**
 * 主布局组件
 * 包含侧边栏导航、顶部栏、WebSocket状态指示器等
 */
const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const { isMobile, sidebarCollapsed: autoCollapsed } = useResponsive();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileDrawerVisible, setMobileDrawerVisible] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  // 自动连接WebSocket
  useWebSocketConnection(true);

  // 响应式布局控制
  useEffect(() => {
    const newCollapsed = isMobile ? true : autoCollapsed;
    // 增加判断，只有在状态需要改变时才更新，防止无限循环
    if (collapsed !== newCollapsed) {
      setCollapsed(newCollapsed);
    }
  }, [isMobile, autoCollapsed, collapsed]);

  // 导航菜单配置
  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: '数据概览',
    },
    {
      key: '/backtest',
      icon: <ExperimentOutlined />,
      label: '回测工作台',
    },
    {
      key: '/strategy',
      icon: <BarsOutlined />,
      label: '策略管理',
    },
    {
      key: '/ai',
      icon: <RobotOutlined />,
      label: 'AI助手',
    },
    {
      key: '/trading',
      icon: <LineChartOutlined />,
      label: '纸上交易',
    },
    {
      key: '/data-center',
      icon: <DatabaseOutlined />,
      label: '数据中心',
    },
  ];

  // 获取当前页面标题 - unused, removed

  // 获取面包屑
  const getBreadcrumbItems = () => {
    const pathMap: Record<string, string> = {
      '/dashboard': '数据概览',
      '/backtest': '回测工作台',
      '/strategy': '策略管理',
      '/ai': 'AI助手',
      '/trading': '纸上交易',
      '/data-center': '数据中心',
    };

    const items = [
      {
        title: <HomeOutlined />,
      },
    ];

    const currentPath = pathMap[location.pathname];
    if (currentPath) {
      items.push({
        title: <span>{currentPath}</span>,
      });
    }

    return items;
  };

  // 菜单点击处理
  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key);
    setMobileDrawerVisible(false);
  };

  // 渲染侧边栏
  const renderSider = (isMobile = false) => (
    <div
      style={{
        height: '100%',
        borderRight: isMobile ? 'none' : '1px solid #f0f0f0',
      }}
    >
      {/* Logo区域 */}
      <div
        style={{
          height: '64px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed && !isMobile ? 'center' : 'flex-start',
          padding: '0 16px',
          borderBottom: '1px solid #f0f0f0',
        }}
      >
        <Space>
          <LineChartOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
          {(!collapsed || isMobile) && (
            <Title level={4} style={{ margin: 0, color: '#1890ff' }}>
              AI量化
            </Title>
          )}
        </Space>
      </div>

      {/* 导航菜单 */}
      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        style={{ border: 'none' }}
        inlineCollapsed={collapsed && !isMobile}
      />
    </div>
  );

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* 桌面端侧边栏 */}
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        breakpoint="lg"
        collapsedWidth={80}
        width={240}
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          zIndex: 100,
          display: isMobile ? 'none' : 'block',
        }}
        className="desktop-sider"
      >
        {renderSider()}
      </Sider>

      {/* 移动端抽屉 */}
      <Drawer
        title={
          <Space>
            <LineChartOutlined style={{ color: '#1890ff' }} />
            <span style={{ color: '#1890ff' }}>AI量化系统</span>
          </Space>
        }
        placement="left"
        onClose={() => setMobileDrawerVisible(false)}
        open={mobileDrawerVisible}
        bodyStyle={{ padding: 0 }}
        width={240}
      >
        {renderSider(true)}
      </Drawer>

      {/* 主内容区域 */}
      <Layout
        style={{
          marginLeft: isMobile ? 0 : collapsed ? 80 : 240,
          transition: 'margin-left 0.2s',
        }}
        className="main-layout"
      >
        {/* 顶部栏 */}
        <Header
          style={{
            background: '#fff',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: isMobile ? '0 16px' : '0 24px',
            position: 'sticky',
            top: 0,
            zIndex: 50,
          }}
        >
          <Space>
            {/* 移动端菜单按钮 */}
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={() => setMobileDrawerVisible(true)}
              className="mobile-menu-button"
              style={{ display: 'none' }}
            />

            {/* 面包屑 */}
            <Breadcrumb items={getBreadcrumbItems()} />
          </Space>

          {/* 右侧工具栏 */}
          <Space>
            <WebSocketIndicator showNotifications />
          </Space>
        </Header>

        {/* 页面内容 */}
        <Content
          style={{
            margin: 0,
            minHeight: 'calc(100vh - 64px)',
            background: '#f5f5f5',
          }}
        >
          {children}
        </Content>
      </Layout>

      {/* 响应式样式 - 移除styled-jsx，使用CSS-in-JS替代 */}
    </Layout>
  );
};

export default MainLayout;
