/**
 * API缓存管理器
 *
 * 提供智能缓存、请求去重、性能优化等功能
 */

interface CacheItem<T> {
  data: T;
  timestamp: number;
  ttl: number; // 生存时间 (ms)
}

interface RequestQueue {
  promise: Promise<any>;
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
}

class APICache {
  private cache = new Map<string, CacheItem<any>>();
  private pendingRequests = new Map<string, RequestQueue[]>();
  private defaultTTL = 5 * 60 * 1000; // 5分钟默认缓存时间

  /**
   * 生成缓存键
   */
  private generateKey(url: string, params?: any): string {
    const paramString = params ? JSON.stringify(params) : '';
    return `${url}:${paramString}`;
  }

  /**
   * 检查缓存是否有效
   */
  private isValidCache<T>(item: CacheItem<T>): boolean {
    return Date.now() - item.timestamp < item.ttl;
  }

  /**
   * 获取缓存数据
   */
  get<T>(url: string, params?: any): T | null {
    const key = this.generateKey(url, params);
    const item = this.cache.get(key);

    if (item && this.isValidCache(item)) {
      return item.data as T;
    }

    // 清理过期缓存
    if (item) {
      this.cache.delete(key);
    }

    return null;
  }

  /**
   * 设置缓存数据
   */
  set<T>(url: string, data: T, params?: any, ttl?: number): void {
    const key = this.generateKey(url, params);
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttl || this.defaultTTL,
    });
  }

  /**
   * 清理指定缓存
   */
  delete(url: string, params?: any): void {
    const key = this.generateKey(url, params);
    this.cache.delete(key);
  }

  /**
   * 清理所有缓存
   */
  clear(): void {
    this.cache.clear();
    this.pendingRequests.clear();
  }

  /**
   * 清理过期缓存
   */
  cleanup(): void {
    for (const [key, item] of this.cache.entries()) {
      if (!this.isValidCache(item)) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * 获取缓存统计信息
   */
  getStats() {
    return {
      totalItems: this.cache.size,
      pendingRequests: this.pendingRequests.size,
      memoryUsage: this.estimateMemoryUsage(),
    };
  }

  /**
   * 估算内存使用
   */
  private estimateMemoryUsage(): number {
    let size = 0;
    for (const [key, item] of this.cache.entries()) {
      size += key.length * 2; // 字符串占用
      size += JSON.stringify(item.data).length * 2; // 数据占用
      size += 24; // 时间戳和TTL占用
    }
    return size;
  }

  /**
   * 请求去重 - 相同请求只执行一次
   */
  async dedupe<T>(key: string, requestFn: () => Promise<T>): Promise<T> {
    // 如果已有相同请求在进行中，等待其完成
    if (this.pendingRequests.has(key)) {
      return new Promise((resolve, reject) => {
        this.pendingRequests
          .get(key)!
          .push({ promise: Promise.resolve(), resolve, reject });
      });
    }

    // 创建新的请求队列
    const requestQueue: RequestQueue[] = [];
    this.pendingRequests.set(key, requestQueue);

    try {
      const result = await requestFn();

      // 通知所有等待的请求
      requestQueue.forEach(({ resolve }) => resolve(result));

      return result;
    } catch (error) {
      // 通知所有等待的请求
      requestQueue.forEach(({ reject }) => reject(error));
      throw error;
    } finally {
      // 清理请求队列
      this.pendingRequests.delete(key);
    }
  }
}

// 全局缓存实例
export const apiCache = new APICache();

/**
 * 缓存装饰器工厂
 */
export function withCache<T extends (...args: any[]) => Promise<any>>(
  ttl?: number,
  keyGenerator?: (...args: Parameters<T>) => string
) {
  return function (
    target: any,
    propertyName: string,
    descriptor: PropertyDescriptor
  ) {
    const method = descriptor.value;

    descriptor.value = async function (...args: Parameters<T>) {
      const cacheKey = keyGenerator
        ? keyGenerator(...args)
        : `${propertyName}:${JSON.stringify(args)}`;

      // 尝试从缓存获取
      const cached = apiCache.get(cacheKey);
      if (cached !== null) {
        return cached;
      }

      // 执行原方法并缓存结果
      const result = await method.apply(this, args);
      apiCache.set(cacheKey, result, undefined, ttl);

      return result;
    };
  };
}

/**
 * 性能优化的请求管理器
 */
class RequestManager {
  private maxConcurrentRequests = 6; // 最大并发请求数
  private activeRequests = new Set<Promise<any>>();
  private requestQueue: Array<() => Promise<any>> = [];

  /**
   * 添加请求到队列
   */
  async queue<T>(requestFn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      const execute = async () => {
        try {
          const promise = requestFn();
          this.activeRequests.add(promise);

          const result = await promise;
          resolve(result);
        } catch (error) {
          reject(error);
        } finally {
          this.activeRequests.delete(promise);
          this.processQueue();
        }
      };

      if (this.activeRequests.size < this.maxConcurrentRequests) {
        execute();
      } else {
        this.requestQueue.push(execute);
      }
    });
  }

  /**
   * 处理队列中的请求
   */
  private processQueue(): void {
    if (
      this.requestQueue.length > 0 &&
      this.activeRequests.size < this.maxConcurrentRequests
    ) {
      const nextRequest = this.requestQueue.shift();
      if (nextRequest) {
        nextRequest();
      }
    }
  }

  /**
   * 获取请求统计
   */
  getStats() {
    return {
      activeRequests: this.activeRequests.size,
      queuedRequests: this.requestQueue.length,
      maxConcurrency: this.maxConcurrentRequests,
    };
  }
}

export const requestManager = new RequestManager();

/**
 * 定期清理缓存
 */
setInterval(
  () => {
    apiCache.cleanup();
  },
  10 * 60 * 1000
); // 每10分钟清理一次

export default apiCache;
