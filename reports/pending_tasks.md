# 待完成工作清单

## 数据适配与存储层
- [x] `AKShareAdapter` 目前的 `get_stock_daily_data` 返回 `pandas.DataFrame`，但端到端与性能测试都期望其产出 `StockData`/`StockDailyBar` 对象集合，导致 `len(stock_data.bars)` 等断言失败。需要提供转换为领域模型的接口或补全 `get_stock_data` 方法以适配 `DataService` 与测试用例。 【F:backend/src/data/akshare_adapter.py†L63-L166】【F:tests/integration/test_end_to_end.py†L55-L112】【F:tests/performance/test_load.py†L53-L115】
- [x] `ParquetStorage` 已提供读写能力，但 `AKShareAdapter` 和 `DataService` 缺少联动，导致端到端流程无法写入/读取真实数据。同时性能测试期望成功率 ≥80%/95%，目前因上述适配问题导致成功率为 0，需要验证存储链路并准备离线样本数据。 【F:backend/app/services/data_service.py†L30-L213】【F:tests/performance/test_load.py†L83-L209】

## 数据同步 API
- [x] `DataService.query_stock_data` 构造 `StockDailyData` 时使用了不存在的 `date` 字段名，FastAPI 在返回 `StockQueryResponse` 时触发验证错误。需改为 `trade_date` 并确认字段完整性。 【F:backend/app/services/data_service.py†L169-L205】【F:backend/app/schemas/data_schemas.py†L88-L131】【81587e†L86-L115】
- [x] `DataService._fetch_stock_data` 调用 `AKShareAdapter.get_stock_data`，但适配器未实现该方法，导致同步任务无法拉取真实数据。需要实现统一入口并处理超时、降级逻辑。 【F:backend/app/services/data_service.py†L214-L251】【F:backend/src/data/akshare_adapter.py†L63-L166】
- [x] `create_sync_task` 等接口的成功路径尚未覆盖批量/单只股票查询测试场景，`tests/test_data_api.py` 中的请求仍返回 500。需完善依赖注入默认实现，确保 Mock 替换后仍可返回符合模式的结构。 【F:backend/app/api/data_sync.py†L96-L213】【81587e†L116-L204】

## 中间件与日志
- [x] 请求跟踪中间件的错误日志模板引用了未提供的 `type` 键，触发 `KeyError` 并掩盖真实异常。需要调整 `logger.error` 的格式化参数。 【81587e†L118-L160】【F:backend/app/middleware/request_tracking.py†L33-L76】

## AI 助手功能
- [x] AI 模块多个端点仍返回占位数据（聊天历史、市场/策略/个股/因子分析、统计信息等），需要接入真实数据源或最少实现可测的存根逻辑。 【F:backend/app/api/ai.py†L470-L661】

## 测试配套
- [x] `tests/integration/test_end_to_end.py` 与 `tests/performance/test_load.py` 内部 `setup_test_environment` fixture 使用实例属性但未与 pytest 机制兼容，当前环境下无法注入 `akshare_adapter`/`parquet_storage`。修复后再运行可验证链路。 【F:tests/integration/test_end_to_end.py†L26-L61】【F:tests/performance/test_load.py†L36-L66】
- [x] 补充覆盖率：现有 `pytest -q` 失败，需在上述功能补全后重新运行直至通过。 【81587e†L205-L214】

> 以上事项按照端到端数据链路 → API → 辅助功能的优先级排序，建议优先完成数据获取/存储与 API 响应结构修复，以便恢复核心回测与数据服务能力。
