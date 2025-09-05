-- AI量化系统 开发环境PostgreSQL初始化脚本
-- 插入测试数据和开发配置

-- 插入测试用户
INSERT INTO users (username, email, password_hash, full_name, role) VALUES 
    ('developer', 'dev@ai-quant.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6wMtb7bfxi', 'Developer User', 'user'),
    ('analyst', 'analyst@ai-quant.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6wMtb7bfxi', 'Analyst User', 'analyst')
ON CONFLICT (username) DO NOTHING;

-- 插入测试策略
INSERT INTO strategies (name, description, user_id, config) VALUES 
    ('均线策略', '基于移动平均线的简单策略', 2, '{"period_short": 5, "period_long": 20}'),
    ('RSI策略', '基于相对强弱指数的策略', 2, '{"rsi_period": 14, "oversold": 30, "overbought": 70}')
ON CONFLICT DO NOTHING;

-- 插入开发环境配置
INSERT INTO app_configs (config_key, config_value, description) VALUES 
    ('debug_mode', 'true', '开发环境调试模式'),
    ('log_level', '"DEBUG"', '开发环境日志级别'),
    ('test_data_enabled', 'true', '是否启用测试数据'),
    ('mock_external_apis', 'true', '是否模拟外部API')
ON CONFLICT (config_key) DO UPDATE SET 
    config_value = EXCLUDED.config_value,
    updated_at = CURRENT_TIMESTAMP;

-- 插入测试同步任务
INSERT INTO sync_tasks (task_type, status, symbols, progress, total_count, completed_count) VALUES 
    ('stock_data', 'completed', ARRAY['000001', '000002', '600000'], 100.0, 3, 3),
    ('stock_data', 'running', ARRAY['300001', '002001'], 50.0, 2, 1),
    ('market_data', 'pending', ARRAY[], 0.0, 0, 0)
ON CONFLICT DO NOTHING;
