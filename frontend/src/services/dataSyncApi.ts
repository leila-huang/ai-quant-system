// 数据同步相关API

import { apiRequest } from './api';
import type {
  DataSyncRequest,
  DataSyncResponse,
  SyncTaskStatus,
} from '@/types/api';

const createSyncTask = (
  payload: DataSyncRequest
): Promise<DataSyncResponse> => {
  return apiRequest.post('/data-sync/sync', payload);
};

const getSyncTaskStatus = (taskId: string): Promise<SyncTaskStatus> => {
  return apiRequest.get(`/data-sync/sync/${taskId}/status`);
};

const cancelSyncTask = (
  taskId: string
): Promise<{ message: string; task_id: string }> => {
  return apiRequest.delete(`/data-sync/sync/${taskId}`);
};

const getDataHealth = (): Promise<any> => {
  return apiRequest.get('/data-sync/health');
};

const getDataStatistics = (): Promise<any> => {
  return apiRequest.get('/data-sync/statistics');
};

const scheduleBackgroundTask = (params: {
  task_type: 'data_sync' | 'data_cleanup' | 'health_check';
  priority?: number;
  symbols?: string[];
  scheduled_at?: string;
}): Promise<any> => {
  return apiRequest.post('/data-sync/tasks/schedule', params);
};

const getTaskQueueStatus = (): Promise<any> => {
  return apiRequest.get('/data-sync/tasks');
};

export const dataSyncApi = {
  createSyncTask,
  getSyncTaskStatus,
  cancelSyncTask,
  getDataHealth,
  getDataStatistics,
  scheduleBackgroundTask,
  getTaskQueueStatus,
};

export default dataSyncApi;
