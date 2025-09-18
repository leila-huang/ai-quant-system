import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // 第三方库代码分割
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'antd-vendor': ['antd', '@ant-design/icons'],
          'chart-vendor': ['echarts', 'echarts-for-react'],
          'utils-vendor': ['axios', 'zustand', 'dayjs'],
        },
      },
    },
    // 增加chunk大小警告限制
    chunkSizeWarningLimit: 1000,
    // 启用source map便于调试
    sourcemap: false, // 生产环境关闭
  },
  // 性能优化
  optimizeDeps: {
    include: ['echarts', 'antd', '@ant-design/icons'],
  },
});
