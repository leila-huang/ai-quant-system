/**
 * 响应式hook - 优化移动端体验
 *
 * 提供屏幕尺寸判断和响应式配置
 */

import { useState, useEffect } from 'react';
import { Grid } from 'antd';

const { useBreakpoint } = Grid;

export interface ResponsiveConfig {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  screenSize: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl';

  // 布局配置
  sidebarCollapsed: boolean;

  // 表格配置
  tableSize: 'small' | 'middle' | 'large';
  tablePagination: {
    pageSize: number;
    showSizeChanger: boolean;
    showQuickJumper: boolean;
    showTotal: boolean;
  };

  // 图表配置
  chartHeight: number;
  chartConfig: {
    responsive: boolean;
    maintainAspectRatio: boolean;
  };

  // 表单配置
  formLayout: 'vertical' | 'horizontal' | 'inline';
  formItemSpan: number;
}

/**
 * 响应式配置hook
 */
export const useResponsive = (): ResponsiveConfig => {
  const screens = useBreakpoint();
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 判断设备类型
  const isMobile = windowWidth < 768 || screens.xs;
  const isTablet =
    (windowWidth >= 768 && windowWidth < 1024) || (screens.sm && !screens.lg);
  const isDesktop = windowWidth >= 1024 || screens.lg;

  // 确定屏幕尺寸
  const getScreenSize = (): ResponsiveConfig['screenSize'] => {
    if (screens.xxl) return 'xxl';
    if (screens.xl) return 'xl';
    if (screens.lg) return 'lg';
    if (screens.md) return 'md';
    if (screens.sm) return 'sm';
    return 'xs';
  };

  const screenSize = getScreenSize();

  // 布局配置
  const sidebarCollapsed = isMobile || isTablet;

  // 表格配置
  const tableSize = isMobile ? 'small' : isTablet ? 'middle' : 'large';
  const tablePagination = {
    pageSize: isMobile ? 5 : isTablet ? 10 : 20,
    showSizeChanger: !isMobile,
    showQuickJumper: isDesktop,
    showTotal: isDesktop,
  };

  // 图表配置
  const chartHeight = isMobile ? 250 : isTablet ? 300 : 400;
  const chartConfig = {
    responsive: true,
    maintainAspectRatio: isMobile,
  };

  // 表单配置
  const formLayout = isMobile ? 'vertical' : 'horizontal';
  const formItemSpan = isMobile ? 24 : isTablet ? 12 : 8;

  return {
    isMobile,
    isTablet,
    isDesktop,
    screenSize,
    sidebarCollapsed,
    tableSize,
    tablePagination,
    chartHeight,
    chartConfig,
    formLayout,
    formItemSpan,
  };
};

/**
 * 设备类型判断hook
 */
export const useDeviceType = () => {
  const { isMobile, isTablet, isDesktop } = useResponsive();

  return {
    isMobile,
    isTablet,
    isDesktop,
    deviceType: isMobile ? 'mobile' : isTablet ? 'tablet' : 'desktop',
  };
};

/**
 * 移动端优化配置
 */
export const useMobileOptimization = () => {
  const { isMobile } = useResponsive();
  const [touchEnabled, setTouchEnabled] = useState(false);

  useEffect(() => {
    // 检测是否支持触摸
    setTouchEnabled('ontouchstart' in window || navigator.maxTouchPoints > 0);
  }, []);

  return {
    isMobile,
    touchEnabled,

    // 移动端特殊配置
    mobileConfig: {
      // 增大点击区域
      minTouchTarget: '44px',

      // 禁用某些动画以提升性能
      animation: !isMobile,

      // 简化UI元素
      showTooltips: !isMobile,
      showSecondaryActions: !isMobile,

      // 移动端专用样式
      padding: isMobile ? '12px' : '24px',
      fontSize: isMobile ? '14px' : '16px',
      lineHeight: isMobile ? '1.4' : '1.6',
    },
  };
};

export default useResponsive;
