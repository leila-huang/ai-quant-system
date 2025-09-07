#!/usr/bin/env python3
"""
技术指标计算引擎验证脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta
import pandas as pd
import numpy as np

from backend.src.models.basic_models import StockData, StockDailyBar
from backend.src.engine.features.indicators import TechnicalIndicators, calculate_single_indicator


def create_test_data():
    """创建测试数据"""
    base_date = date(2024, 1, 1)
    test_bars = []
    
    # 创建一个简单的上涨趋势数据
    prices = [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 12.0, 11.9,
             12.2, 12.5, 12.3, 12.8, 13.0, 12.8, 13.2, 13.5, 13.3, 13.8]
    volumes = [1000, 1100, 1200, 1050, 1300, 1150, 1400, 1250, 1350, 1200,
               1450, 1300, 1380, 1420, 1500, 1350, 1480, 1550, 1420, 1600]
    
    for i, (price, volume) in enumerate(zip(prices, volumes)):
        bar = StockDailyBar(
            date=base_date + timedelta(days=i),
            open_price=price - 0.1,
            high_price=price + 0.2,
            low_price=price - 0.2,
            close_price=price,
            volume=volume,
            amount=price * volume * 100,
            adjust_factor=1.0
        )
        test_bars.append(bar)
    
    return StockData(
        symbol="000001",
        name="测试股票",
        bars=test_bars
    )


def test_basic_functionality():
    """测试基本功能"""
    print("🔍 测试基本功能...")
    
    # 创建测试数据
    test_data = create_test_data()
    
    # 创建计算器
    calculator = TechnicalIndicators()
    
    # 测试StockData转DataFrame
    df = test_data.to_dataframe()
    print(f"✅ StockData转DataFrame成功: {df.shape}")
    
    # 测试支持的指标列表
    indicators = calculator.get_supported_indicators()
    print(f"✅ 支持的指标数量: {len(indicators)}")
    print(f"   指标列表: {indicators}")
    
    return test_data, calculator


def test_individual_indicators():
    """测试各个指标的计算"""
    print("\n📊 测试各个技术指标...")
    
    test_data, calculator = test_basic_functionality()
    
    try:
        # 测试移动平均线
        print("测试 MA (移动平均线)...")
        ma_result = calculate_single_indicator(test_data, "MA", ma_windows=[5, 10, 20])
        print(f"✅ MA计算成功，新增列数: {len([col for col in ma_result.columns if 'ma_' in col])}")
        
        # 测试RSI
        print("测试 RSI (相对强弱指标)...")
        rsi_result = calculate_single_indicator(test_data, "RSI", rsi_windows=[14])
        rsi_value = rsi_result['rsi_14'].iloc[-1]
        print(f"✅ RSI计算成功，最新RSI值: {rsi_value:.2f}")
        
        # 测试MACD
        print("测试 MACD...")
        macd_result = calculate_single_indicator(test_data, "MACD")
        macd_value = macd_result['macd'].iloc[-1]
        print(f"✅ MACD计算成功，最新MACD值: {macd_value:.4f}")
        
        # 测试布林带
        print("测试 BOLL (布林带)...")
        boll_result = calculate_single_indicator(test_data, "BOLL")
        upper = boll_result['boll_upper'].iloc[-1]
        lower = boll_result['boll_lower'].iloc[-1] 
        print(f"✅ 布林带计算成功，上轨: {upper:.2f}, 下轨: {lower:.2f}")
        
        # 测试KDJ
        print("测试 KDJ...")
        kdj_result = calculate_single_indicator(test_data, "KDJ")
        k_value = kdj_result['kdj_k'].iloc[-1]
        d_value = kdj_result['kdj_d'].iloc[-1]
        print(f"✅ KDJ计算成功，K值: {k_value:.2f}, D值: {d_value:.2f}")
        
    except Exception as e:
        print(f"❌ 个别指标测试失败: {e}")
        return False
    
    return True


def test_all_indicators():
    """测试计算所有指标"""
    print("\n🔄 测试计算所有指标...")
    
    test_data, calculator = test_basic_functionality()
    
    try:
        # 计算所有指标
        result = calculator.calculate(test_data)
        
        print(f"✅ 所有指标计算成功")
        print(f"   数据行数: {len(result)}")
        print(f"   总列数: {len(result.columns)}")
        
        # 显示一些关键指标的最新值
        print(f"\n📈 最新指标值样本:")
        if 'ma_5' in result.columns:
            print(f"   5日均线: {result['ma_5'].iloc[-1]:.2f}")
        if 'rsi_14' in result.columns:
            print(f"   RSI(14): {result['rsi_14'].iloc[-1]:.2f}")
        if 'macd' in result.columns:
            print(f"   MACD: {result['macd'].iloc[-1]:.4f}")
        if 'boll_position' in result.columns:
            print(f"   布林带位置: {result['boll_position'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 计算所有指标失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """测试性能"""
    print("\n⚡ 测试计算性能...")
    
    # 创建较大的测试数据集
    base_date = date(2023, 1, 1)
    large_bars = []
    
    # 250个交易日的数据
    np.random.seed(42)
    for i in range(250):
        price = 10.0 + np.sin(i * 0.02) * 2 + np.random.normal(0, 0.1)
        price = max(5.0, price)  # 确保价格为正
        
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
    
    large_data = StockData(symbol="PERF", name="性能测试", bars=large_bars)
    calculator = TechnicalIndicators()
    
    import time
    start_time = time.time()
    
    try:
        result = calculator.calculate(large_data)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        print(f"✅ 性能测试通过")
        print(f"   数据量: {len(large_data.bars)} 条记录")
        print(f"   计算时间: {calculation_time:.3f} 秒")
        print(f"   处理速度: {len(large_data.bars)/calculation_time:.0f} 条/秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


def test_data_validation():
    """测试数据验证"""
    print("\n✅ 测试数据验证...")
    
    calculator = TechnicalIndicators()
    
    # 测试空数据
    empty_data = StockData(symbol="EMPTY", name="空数据", bars=[])
    result = calculator.calculate(empty_data)
    print(f"✅ 空数据处理正确: {result.empty}")
    
    # 测试无效指标名称
    test_data = create_test_data()
    try:
        calculator.calculate(test_data, indicators=["INVALID"])
        print("❌ 应该抛出异常但没有")
        return False
    except ValueError:
        print("✅ 无效指标名称正确抛出异常")
    
    return True


def main():
    """主函数"""
    print("🚀 AI量化系统 - 技术指标计算引擎验证")
    print("=" * 60)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(test_individual_indicators())
    test_results.append(test_all_indicators()) 
    test_results.append(test_performance())
    test_results.append(test_data_validation())
    
    # 总结测试结果
    print("\n" + "=" * 60)
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    if passed_tests == total_tests:
        print(f"🎉 所有测试通过! ({passed_tests}/{total_tests})")
        print("\n📊 技术指标计算引擎验证完成，准备投入使用！")
        return 0
    else:
        print(f"⚠️  部分测试失败: {passed_tests}/{total_tests}")
        return 1


if __name__ == "__main__":
    sys.exit(main())



