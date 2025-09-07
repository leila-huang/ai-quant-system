"""
技术指标计算引擎单元测试

验证各种技术指标计算的准确性和性能。
"""

import unittest
from datetime import date, timedelta
import pandas as pd
import numpy as np

from backend.src.models.basic_models import StockData, StockDailyBar
from backend.src.engine.features.indicators import TechnicalIndicators, calculate_single_indicator


class TestTechnicalIndicators(unittest.TestCase):
    """技术指标测试类"""
    
    def setUp(self):
        """设置测试数据"""
        self.calculator = TechnicalIndicators()
        
        # 创建测试数据 - 20天的模拟股票数据
        base_date = date(2024, 1, 1)
        self.test_bars = []
        
        # 创建一个简单的趋势数据
        prices = [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 12.0, 11.9,
                 12.2, 12.5, 12.3, 12.8, 13.0, 12.8, 13.2, 13.5, 13.3, 13.8]
        volumes = [1000] * 20
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            bar = StockDailyBar(
                date=base_date + timedelta(days=i),
                open_price=price - 0.1,
                high_price=price + 0.2,
                low_price=price - 0.2,
                close_price=price,
                volume=volume * (1 + i * 0.1),  # 递增的成交量
                amount=price * volume * 100,
                adjust_factor=1.0
            )
            self.test_bars.append(bar)
        
        self.test_stock_data = StockData(
            symbol="000001",
            name="测试股票",
            bars=self.test_bars
        )
        
        # 创建测试DataFrame
        self.test_df = self.test_stock_data.to_dataframe()
    
    def test_data_preparation(self):
        """测试数据准备和转换"""
        # 测试StockData转DataFrame
        df = self.test_stock_data.to_dataframe()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 20)
        
        # 验证必需的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        # 验证数据类型
        for col in required_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))
    
    def test_supported_indicators(self):
        """测试支持的指标列表"""
        indicators = self.calculator.get_supported_indicators()
        expected_indicators = ["MA", "EMA", "RSI", "MACD", "BOLL", "KDJ", "WILLIAMS", "VOLUME_MA"]
        
        self.assertEqual(set(indicators), set(expected_indicators))
    
    def test_ma_calculation(self):
        """测试移动平均线计算"""
        result = calculate_single_indicator(self.test_stock_data, "MA", ma_windows=[5, 10])
        
        # 验证MA列存在
        self.assertIn('ma_5', result.columns)
        self.assertIn('ma_10', result.columns)
        self.assertIn('close_ma_5_ratio', result.columns)
        
        # 验证MA计算准确性（手动验证前几个值）
        close_prices = result['close'].values
        ma_5_manual = pd.Series(close_prices).rolling(5, min_periods=1).mean()
        
        # 允许小的浮点误差
        np.testing.assert_array_almost_equal(
            result['ma_5'].values, ma_5_manual.values, decimal=6
        )
    
    def test_rsi_calculation(self):
        """测试RSI指标计算"""
        result = calculate_single_indicator(self.test_stock_data, "RSI", rsi_windows=[14])
        
        # 验证RSI列存在
        self.assertIn('rsi_14', result.columns)
        
        # 验证RSI值在0-100范围内
        rsi_values = result['rsi_14'].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_values))
        
        # RSI应该不是NaN（除了第一个值）
        self.assertFalse(result['rsi_14'].iloc[-1] is np.nan)
    
    def test_macd_calculation(self):
        """测试MACD指标计算"""
        result = calculate_single_indicator(self.test_stock_data, "MACD")
        
        # 验证MACD相关列存在
        macd_cols = ['macd', 'macd_signal', 'macd_hist']
        for col in macd_cols:
            self.assertIn(col, result.columns)
        
        # 验证MACD柱状图 = MACD - Signal
        np.testing.assert_array_almost_equal(
            result['macd_hist'].values,
            (result['macd'] - result['macd_signal']).values,
            decimal=10
        )
    
    def test_bollinger_bands_calculation(self):
        """测试布林带指标计算"""
        result = calculate_single_indicator(self.test_stock_data, "BOLL")
        
        # 验证布林带列存在
        boll_cols = ['boll_upper', 'boll_middle', 'boll_lower', 'boll_width', 'boll_position']
        for col in boll_cols:
            self.assertIn(col, result.columns)
        
        # 验证上轨 > 中轨 > 下轨
        for i in range(len(result)):
            if not pd.isna(result['boll_upper'].iloc[i]):
                self.assertGreater(result['boll_upper'].iloc[i], result['boll_middle'].iloc[i])
                self.assertGreater(result['boll_middle'].iloc[i], result['boll_lower'].iloc[i])
        
        # 验证位置指标在0-1范围内（大部分情况）
        positions = result['boll_position'].dropna()
        # 允许少量超出范围的情况（正常的市场行为）
        within_range = positions.between(-0.5, 1.5).mean()
        self.assertGreater(within_range, 0.8)  # 80%以上在合理范围内
    
    def test_kdj_calculation(self):
        """测试KDJ指标计算"""
        result = calculate_single_indicator(self.test_stock_data, "KDJ")
        
        # 验证KDJ列存在
        kdj_cols = ['kdj_k', 'kdj_d', 'kdj_j']
        for col in kdj_cols:
            self.assertIn(col, result.columns)
        
        # 验证KDJ值在合理范围内（0-100，J线可能超出）
        k_values = result['kdj_k'].dropna()
        d_values = result['kdj_d'].dropna()
        
        self.assertTrue(all(0 <= val <= 100 for val in k_values))
        self.assertTrue(all(0 <= val <= 100 for val in d_values))
    
    def test_williams_r_calculation(self):
        """测试威廉指标计算"""
        result = calculate_single_indicator(self.test_stock_data, "WILLIAMS")
        
        # 验证威廉指标列存在
        self.assertIn('williams_r', result.columns)
        
        # 验证威廉指标值在-100到0范围内
        williams_values = result['williams_r'].dropna()
        self.assertTrue(all(-100 <= val <= 0 for val in williams_values))
    
    def test_volume_ma_calculation(self):
        """测试成交量移动平均计算"""
        result = calculate_single_indicator(self.test_stock_data, "VOLUME_MA")
        
        # 验证成交量相关列存在
        volume_cols = ['volume_ma_5', 'volume_ma_10', 'volume_ratio']
        for col in volume_cols:
            self.assertIn(col, result.columns)
        
        # 验证成交量比率大于0
        volume_ratios = result['volume_ratio'].dropna()
        self.assertTrue(all(val > 0 for val in volume_ratios))
    
    def test_all_indicators_calculation(self):
        """测试计算所有指标"""
        result = self.calculator.calculate(self.test_stock_data)
        
        # 验证返回的DataFrame不为空
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 20)
        
        # 验证包含原始数据列
        original_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in original_cols:
            self.assertIn(col, result.columns)
        
        # 验证包含技术指标列
        # 至少应该有一些技术指标列
        indicator_cols = [col for col in result.columns if col not in original_cols + ['symbol', 'name', 'amount', 'adjust_factor']]
        self.assertGreater(len(indicator_cols), 10)  # 应该有超过10个技术指标
    
    def test_invalid_data_handling(self):
        """测试无效数据处理"""
        # 测试空数据
        empty_stock_data = StockData(symbol="EMPTY", name="空数据", bars=[])
        result = self.calculator.calculate(empty_stock_data)
        self.assertTrue(result.empty)
        
        # 测试不支持的指标
        with self.assertRaises(ValueError):
            self.calculator.calculate(self.test_stock_data, indicators=["INVALID_INDICATOR"])
    
    def test_performance_with_large_data(self):
        """测试大数据集的性能"""
        # 创建大数据集（1000个交易日）
        base_date = date(2020, 1, 1)
        large_bars = []
        
        for i in range(1000):
            price = 10.0 + np.sin(i * 0.01) * 2 + np.random.normal(0, 0.1)
            bar = StockDailyBar(
                date=base_date + timedelta(days=i),
                open_price=price - 0.1,
                high_price=price + 0.2,
                low_price=price - 0.2,
                close_price=price,
                volume=1000 + np.random.randint(-100, 100),
                amount=price * 1000 * 100
            )
            large_bars.append(bar)
        
        large_stock_data = StockData(symbol="LARGE", name="大数据测试", bars=large_bars)
        
        import time
        start_time = time.time()
        result = self.calculator.calculate(large_stock_data)
        end_time = time.time()
        
        # 验证计算完成
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 1000)
        
        # 计算时间应该在合理范围内（<5秒）
        calculation_time = end_time - start_time
        self.assertLess(calculation_time, 5.0, f"计算时间过长: {calculation_time:.2f}秒")
        
        print(f"大数据集（1000条记录）计算时间: {calculation_time:.3f}秒")
    
    def test_feature_names(self):
        """测试特征名称列表"""
        feature_names = self.calculator.get_feature_names()
        
        # 验证返回的是列表
        self.assertIsInstance(feature_names, list)
        
        # 验证不为空
        self.assertGreater(len(feature_names), 0)
        
        # 验证包含预期的特征名
        expected_names = ['ma_5', 'ma_10', 'rsi_14', 'macd', 'boll_upper']
        for name in expected_names:
            self.assertIn(name, feature_names)
    
    def test_dataframe_input(self):
        """测试DataFrame输入"""
        result = self.calculator.calculate(self.test_df, indicators=["MA"])
        
        # 验证能够处理DataFrame输入
        self.assertFalse(result.empty)
        self.assertIn('ma_5', result.columns)


if __name__ == '__main__':
    unittest.main()



