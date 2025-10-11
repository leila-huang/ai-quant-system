import { describe, test, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor } from '@testing-library/react';
import QuickActions from '@/components/QuickActions';
import * as services from '@/services';

vi.mock('@/services', () => {
  return {
    apiRequest: {
      get: vi.fn(),
      post: vi.fn(),
    },
    dataApi: {
      getStatus: vi.fn(),
      createSampleData: vi.fn(),
    },
    dataSyncApi: {
      createSyncTask: vi.fn(),
      getSyncTaskStatus: vi.fn(),
    },
  } as any;
});

describe('QuickActions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test('点击样本数据按钮应调用创建样本数据API', async () => {
    vi.mocked(services.dataApi.createSampleData as any).mockResolvedValue({
      symbol: 'SAMPLE001',
      bars_count: 30,
    });

    const onRefreshData = vi.fn();
    const { getByText } = render(
      <QuickActions onRefreshData={onRefreshData} />
    );

    fireEvent.click(getByText('样本数据'));

    await waitFor(() => {
      expect(services.dataApi.createSampleData).toHaveBeenCalled();
      expect(onRefreshData).toHaveBeenCalled();
    });
  });

  test('点击数据同步应创建任务并轮询一次', async () => {
    vi.mocked(services.dataSyncApi.createSyncTask as any).mockResolvedValue({
      task_id: 'task1',
    });
    vi.mocked(services.dataSyncApi.getSyncTaskStatus as any).mockResolvedValue({
      status: 'success',
      progress: 100,
      symbols_completed: 1,
    });

    const { getByText } = render(<QuickActions />);

    fireEvent.click(getByText('数据同步'));
    // 确认弹窗
    await waitFor(() => {
      const ok = document.querySelector(
        '.ant-btn-primary'
      ) as HTMLButtonElement;
      ok?.click();
    });

    await waitFor(() => {
      expect(services.dataSyncApi.createSyncTask).toHaveBeenCalled();
      expect(services.dataSyncApi.getSyncTaskStatus).toHaveBeenCalled();
    });
  });
});
