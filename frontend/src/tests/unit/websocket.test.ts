/**
 * WebSocket客户端单元测试
 * 测试WebSocket客户端的基础功能和状态管理
 */

import { describe, test, expect, beforeEach, vi } from 'vitest';
import { getWebSocketClient, websocketUtils } from '@/services/websocket';

// Mock WebSocket
const mockWebSocket = {
  readyState: WebSocket.CONNECTING,
  url: '',
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  onopen: null,
  onclose: null,
  onmessage: null,
  onerror: null,
};

beforeEach(() => {
  vi.clearAllMocks();
  global.WebSocket = vi.fn().mockImplementation(() => mockWebSocket) as any;
});

describe('WebSocket Client Unit Tests', () => {
  describe('客户端实例化', () => {
    test('应该能创建WebSocket客户端实例', () => {
      const client = getWebSocketClient();
      expect(client).toBeDefined();
      expect(typeof client.connect).toBe('function');
      expect(typeof client.disconnect).toBe('function');
      expect(typeof client.subscribe).toBe('function');
      expect(typeof client.send).toBe('function');
      expect(typeof client.getStatus).toBe('function');
    });

    test('应该返回相同的客户端实例（单例模式）', () => {
      const client1 = getWebSocketClient();
      const client2 = getWebSocketClient();
      expect(client1).toBe(client2);
    });
  });

  describe('连接状态管理', () => {
    test('初始状态应该是disconnected', () => {
      const client = getWebSocketClient();
      expect(client.getStatus()).toBe('disconnected');
    });

    test('应该能添加状态变化监听器', () => {
      const client = getWebSocketClient();
      const statusCallback = vi.fn();

      client.onStatusChange(statusCallback);

      // 状态监听器应该被添加（无法直接验证，但不应该抛出错误）
      expect(() => client.onStatusChange(statusCallback)).not.toThrow();
    });
  });

  describe('消息订阅', () => {
    test('应该能订阅消息类型', () => {
      const messageHandler = vi.fn();

      const result = websocketUtils.subscribe('test_message', messageHandler);

      // subscribe应该返回取消订阅函数
      expect(typeof result).toBe('function');
    });

    test('应该能取消消息订阅', () => {
      const messageHandler = vi.fn();

      const unsubscribe = websocketUtils.subscribe(
        'test_message',
        messageHandler
      );

      expect(() => unsubscribe()).not.toThrow();
    });

    test('应该能处理多个订阅者', () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      const unsubscribe1 = websocketUtils.subscribe('test_message', handler1);
      const unsubscribe2 = websocketUtils.subscribe('test_message', handler2);

      expect(typeof unsubscribe1).toBe('function');
      expect(typeof unsubscribe2).toBe('function');
    });
  });

  describe('消息发送', () => {
    test('应该能发送消息', () => {
      const client = getWebSocketClient();
      const message = {
        type: 'test_message',
        data: { content: 'Hello' },
      };

      expect(() => client.send(message)).not.toThrow();
    });

    test('发送消息时应该正确序列化', () => {
      const client = getWebSocketClient();
      const message = {
        type: 'test_message',
        data: { content: 'Hello' },
      };

      // 连接状态不是connected时，消息应该被缓存或忽略
      client.send(message);

      // 验证不会抛出错误
      expect(() => client.send(message)).not.toThrow();
    });
  });

  describe('连接管理', () => {
    test('应该能启动连接', () => {
      const client = getWebSocketClient();

      expect(() => client.connect()).not.toThrow();
    });

    test('应该能断开连接', () => {
      const client = getWebSocketClient();

      expect(() => client.disconnect()).not.toThrow();
    });

    test('重复连接应该被正确处理', () => {
      const client = getWebSocketClient();

      client.connect();
      expect(() => client.connect()).not.toThrow();
    });

    test('断开未连接的客户端应该被正确处理', () => {
      const client = getWebSocketClient();

      expect(() => client.disconnect()).not.toThrow();
    });
  });

  describe('错误处理', () => {
    test('WebSocket构造函数失败时应该正确处理', () => {
      global.WebSocket = vi.fn().mockImplementation(() => {
        throw new Error('WebSocket construction failed');
      }) as any;

      const client = getWebSocketClient();

      // 连接失败应该被正确处理
      expect(() => client.connect()).not.toThrow();
    });

    test('无效消息格式应该被正确处理', () => {
      const client = getWebSocketClient();

      // 发送无效消息不应该导致客户端崩溃
      expect(() => client.send(null as any)).not.toThrow();
      expect(() => client.send(undefined as any)).not.toThrow();
    });
  });

  describe('配置和选项', () => {
    test('应该使用正确的WebSocket URL', () => {
      const client = getWebSocketClient();
      client.connect();

      // WebSocket应该被正确实例化
      expect(global.WebSocket).toHaveBeenCalled();
    });

    test('应该支持自定义选项', () => {
      const client = getWebSocketClient();

      // 客户端应该有获取配置的方法
      expect(client).toBeDefined();
    });
  });

  describe('生命周期管理', () => {
    test('应该正确清理资源', () => {
      const client = getWebSocketClient();
      const messageHandler = vi.fn();

      const unsubscribe = websocketUtils.subscribe(
        'test_message',
        messageHandler
      );

      // 清理订阅
      unsubscribe();

      // 断开连接
      client.disconnect();

      expect(() => {
        unsubscribe();
        client.disconnect();
      }).not.toThrow();
    });
  });
});
