#!/usr/bin/env python3
"""
使用真实AKShare数据验证技术指标计算引擎
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta
import time

from backend.src.data.akshare_adapter import AKShareAdapter
from backend.src.engine.features.indicators import TechnicalIndicators


def test_with_real_data():
    """使用真实AKShare数据测试"""
    print("🔍 使用真实AKShare数据验证技术指标计算...")
    
    try:
        # 创建数据适配器
        adapter = AKShareAdapter()
        
        # 获取平安银行近3个月数据
        end_date = date.today()
        start_date = end_date - timedelta(days=90)
        
        print(f"正在获取平安银行(000001)数据: {start_date} 到 {end_date}")
        stock_data = adapter.get_stock_data("000001", start_date, end_date)
        
        if not stock_data or not stock_data.bars:
            print("❌ 未能获取到数据")
            return False
        
        print(f"✅ 成功获取数据，共 {len(stock_data.bars)} 个交易日")
        
        # 创建技术指标计算器
        calculator = TechnicalIndicators()
        
        # 计算技术指标
        print("正在计算技术指标...")
        start_time = time.time()
        
        result = calculator.calculate(stock_data)
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        print(f"✅ 技术指标计算完成")
        print(f"   计算时间: {calculation_time:.3f} 秒")
        print(f"   数据维度: {result.shape}")
        
        # 展示最新的技术指标值
        print(f"\n📊 最新技术指标值 (平安银行 000001):")
        latest = result.iloc[-1]
        
        print(f"   股票代码: {latest.get('symbol', 'N/A')}")
        print(f"   最新日期: {latest.get('date', 'N/A')}")
        print(f"   收盘价: {latest.get('close', 0):.2f}")
        
        # 移动平均线
        if 'ma_5' in result.columns:
            print(f"   5日均线: {latest.get('ma_5', 0):.2f}")
        if 'ma_20' in result.columns:
            print(f"   20日均线: {latest.get('ma_20', 0):.2f}")
            
        # RSI
        if 'rsi_14' in result.columns:
            rsi = latest.get('rsi_14', 0)
            rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "正常"
            print(f"   RSI(14): {rsi:.2f} ({rsi_status})")
            
        # MACD
        if 'macd' in result.columns:
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_trend = "看多" if macd > macd_signal else "看空"
            print(f"   MACD: {macd:.4f} ({macd_trend})")
            
        # 布林带
        if all(col in result.columns for col in ['boll_upper', 'boll_lower', 'boll_position']):
            position = latest.get('boll_position', 0)
            boll_status = "接近上轨" if position > 0.8 else "接近下轨" if position < 0.2 else "中位运行"
            print(f"   布林带位置: {position:.2f} ({boll_status})")
            
        # KDJ
        if all(col in result.columns for col in ['kdj_k', 'kdj_d']):
            k = latest.get('kdj_k', 0)
            d = latest.get('kdj_d', 0)
            kdj_trend = "金叉" if k > d else "死叉"
            print(f"   KDJ: K={k:.1f}, D={d:.1f} ({kdj_trend})")
        
        # 成交量比率
        if 'volume_ratio' in result.columns:
            vol_ratio = latest.get('volume_ratio', 0)
            vol_status = "放量" if vol_ratio > 1.5 else "缩量" if vol_ratio < 0.8 else "正常"
            print(f"   成交量比率: {vol_ratio:.2f} ({vol_status})")
        
        print("\n✅ 真实数据验证成功！")
        return True
        
    except Exception as e:
        print(f"❌ 真实数据验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 AI量化系统 - 真实数据验证")
    print("=" * 60)
    
    success = test_with_real_data()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 真实数据验证通过！技术指标计算引擎可以正常处理AKShare数据。")
        return 0
    else:
        print("⚠️  真实数据验证失败，请检查网络连接或数据源。")
        return 1


if __name__ == "__main__":
    sys.exit(main())



