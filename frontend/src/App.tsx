import { lazy, Suspense } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from 'react-router-dom';
import { ConfigProvider, theme, Spin } from 'antd';
import zhCN from 'antd/locale/zh_CN';

import MainLayout from '@/components/MainLayout';
import PerformanceMonitor from '@/components/PerformanceMonitor';
import ErrorBoundary from '@/components/ErrorBoundary';
import { measureWebVitals } from '@/utils/performance';

// 懒加载页面组件 - 性能优化
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const Backtest = lazy(() => import('@/pages/Backtest'));
const Strategy = lazy(() => import('@/pages/Strategy'));
const AI = lazy(() => import('@/pages/AI'));
const Trading = lazy(() => import('@/pages/Trading'));

// Import Ant Design styles
import 'antd/dist/reset.css';

// Import mobile optimization styles
import '@/styles/mobile.css';

// 初始化性能监控
measureWebVitals();

function App() {
  return (
    <ErrorBoundary>
      <ConfigProvider
        locale={zhCN}
        theme={{
          algorithm: theme.defaultAlgorithm,
          token: {
            colorPrimary: '#1890ff',
            borderRadius: 6,
            colorBgContainer: '#ffffff',
          },
        }}
      >
        <Router>
          <MainLayout>
            <ErrorBoundary>
              <Suspense
                fallback={
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      height: '400px',
                      flexDirection: 'column',
                      gap: '16px',
                    }}
                  >
                    <Spin size="large" />
                    <div style={{ color: '#666', fontSize: '14px' }}>
                      页面加载中...
                    </div>
                  </div>
                }
              >
                <Routes>
                  <Route
                    path="/"
                    element={<Navigate to="/dashboard" replace />}
                  />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/backtest" element={<Backtest />} />
                  <Route path="/strategy" element={<Strategy />} />
                  <Route path="/ai" element={<AI />} />
                  <Route path="/trading" element={<Trading />} />
                </Routes>
              </Suspense>
            </ErrorBoundary>
          </MainLayout>
        </Router>

        {/* 性能监控组件，仅开发环境显示 */}
        <PerformanceMonitor />
      </ConfigProvider>
    </ErrorBoundary>
  );
}

export default App;
