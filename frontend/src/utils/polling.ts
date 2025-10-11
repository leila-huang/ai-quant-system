// 通用轮询工具，遵循SOLID：单一职责（仅负责轮询控制），可通过参数注入依赖
export interface PollOptions<T> {
  fetch: () => Promise<T>; // 依赖注入：获取状态函数
  isDone: (result: T) => boolean; // 结束条件判断
  intervalMs?: number; // 间隔
  timeoutMs?: number; // 超时时间
  onTick?: (result: T, attempt: number) => void; // 每次轮询回调
}

export async function pollUntil<T>(options: PollOptions<T>): Promise<T> {
  const {
    fetch,
    isDone,
    intervalMs = 2000,
    timeoutMs = 120000,
    onTick,
  } = options;

  const start = Date.now();
  let attempt = 0;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    // eslint-disable-next-line no-await-in-loop
    const result = await fetch();
    attempt += 1;
    if (onTick) onTick(result, attempt);
    if (isDone(result)) return result;

    if (Date.now() - start >= timeoutMs) {
      return result; // 返回最后一次结果，交由调用者处理超时
    }

    // eslint-disable-next-line no-await-in-loop
    await new Promise(r => setTimeout(r, intervalMs));
  }
}
