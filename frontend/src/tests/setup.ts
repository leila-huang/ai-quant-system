/**
 * 测试环境设置
 */

import '@testing-library/jest-dom';
import { vi, afterEach } from 'vitest';

// 模拟全局对象和API
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// 模拟ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// 模拟IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
  takeRecords: vi.fn(() => []),
  root: null,
  rootMargin: '0px',
  thresholds: [0],
}));

// 模拟ECharts
vi.mock('echarts', () => ({
  default: {
    init: vi.fn(() => ({
      setOption: vi.fn(),
      resize: vi.fn(),
      dispose: vi.fn(),
      on: vi.fn(),
      off: vi.fn(),
      getOption: vi.fn(),
    })),
    getInstanceByDom: vi.fn(),
    dispose: vi.fn(),
  },
}));

// 模拟echarts-for-react
vi.mock('echarts-for-react', () => ({
  default: vi.fn(
    ({ option, style, className }) =>
      `<div class="echarts-react ${className || ''}" style="${JSON.stringify(style)}" data-option="${JSON.stringify(option)}"></div>`
  ),
}));

// 模拟axios
const mockAxiosInstance = {
  get: vi.fn(() => Promise.resolve({ data: {} })),
  post: vi.fn(() => Promise.resolve({ data: {} })),
  put: vi.fn(() => Promise.resolve({ data: {} })),
  delete: vi.fn(() => Promise.resolve({ data: {} })),
  interceptors: {
    request: { use: vi.fn(), eject: vi.fn() },
    response: { use: vi.fn(), eject: vi.fn() },
  },
};

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => mockAxiosInstance),
    get: vi.fn(() => Promise.resolve({ data: {} })),
    post: vi.fn(() => Promise.resolve({ data: {} })),
    interceptors: {
      request: { use: vi.fn(), eject: vi.fn() },
      response: { use: vi.fn(), eject: vi.fn() },
    },
  },
}));

// 模拟WebSocket
global.WebSocket = vi.fn().mockImplementation((url: string) => ({
  url,
  readyState: WebSocket.CONNECTING,
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  onopen: null,
  onclose: null,
  onmessage: null,
  onerror: null,
}));

// 模拟fetch
global.fetch = vi.fn(() =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
  })
) as any;

// 模拟requestAnimationFrame
global.requestAnimationFrame = vi
  .fn()
  .mockImplementation((callback: FrameRequestCallback) => {
    return setTimeout(callback, 16); // 模拟60fps
  });

global.cancelAnimationFrame = vi.fn().mockImplementation((id: number) => {
  clearTimeout(id);
});

// 模拟console方法以避免测试输出混乱
console.log = vi.fn();
console.info = vi.fn();
console.warn = vi.fn();
console.error = vi.fn();

// 全局错误处理
window.addEventListener('error', event => {
  console.error('Unhandled error during test:', event.error);
});

window.addEventListener('unhandledrejection', event => {
  console.error('Unhandled promise rejection during test:', event.reason);
});

// 每个测试后清理
afterEach(() => {
  vi.clearAllMocks();
});
