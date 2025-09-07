#!/usr/bin/env python3
"""
P1级依赖验证脚本
验证所有P1级新增的依赖包是否能正常导入
"""

import sys
import importlib
from typing import List, Tuple

# P1级新增的依赖包
P1_DEPENDENCIES = [
    ('vectorbt', 'vectorbt'),
    ('xgboost', 'xgboost'),
    ('sklearn', 'scikit-learn'),
    ('talib', 'TA-Lib'),
    ('numba', 'numba'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('joblib', 'joblib'),
]

def verify_dependencies() -> Tuple[List[str], List[str]]:
    """
    验证依赖包导入情况
    
    Returns:
        Tuple[List[str], List[str]]: (成功导入的包, 失败的包)
    """
    successful = []
    failed = []
    
    print("🔍 正在验证P1级依赖包...")
    print("-" * 50)
    
    for module_name, package_name in P1_DEPENDENCIES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package_name:15} - {version}")
            successful.append(package_name)
        except ImportError as e:
            print(f"❌ {package_name:15} - 导入失败: {e}")
            failed.append(package_name)
    
    return successful, failed

def test_basic_functionality():
    """测试核心功能的基本可用性"""
    print("\n🧪 测试基本功能...")
    print("-" * 50)
    
    try:
        # 测试 pandas + numpy 基础操作
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            'close': [10, 11, 12, 11, 13, 12, 14],
            'volume': [1000, 1100, 1200, 1050, 1300, 1150, 1400]
        })
        print(f"✅ 基础数据处理 - DataFrame创建成功: {df.shape}")
        
        # 测试技术指标计算 (简单移动平均)
        df['ma_3'] = df['close'].rolling(3).mean()
        print(f"✅ 简单移动平均 - 计算成功: {df['ma_3'].iloc[-1]:.2f}")
        
        # 测试vectorbt (如果可用)
        try:
            import vectorbt as vbt
            print(f"✅ vectorbt - 版本: {vbt.__version__}")
        except ImportError:
            print("⚠️  vectorbt - 尚未安装，建议执行: pip install vectorbt>=0.25.0")
        
        # 测试XGBoost (如果可用)
        try:
            import xgboost as xgb
            print(f"✅ XGBoost - 版本: {xgb.__version__}")
        except ImportError:
            print("⚠️  XGBoost - 尚未安装，建议执行: pip install xgboost>=2.0.0")
        
        # 测试sklearn
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            print(f"✅ scikit-learn - 基础模块导入成功")
        except ImportError:
            print("⚠️  scikit-learn - 尚未安装，建议执行: pip install scikit-learn>=1.3.0")
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")

def main():
    """主函数"""
    print("🚀 AI量化系统 P1级依赖验证")
    print("=" * 60)
    
    # 验证依赖导入
    successful, failed = verify_dependencies()
    
    # 输出统计信息
    print("\n📊 验证结果统计")
    print("-" * 50)
    print(f"✅ 成功导入: {len(successful)}/{len(P1_DEPENDENCIES)} 个包")
    print(f"❌ 导入失败: {len(failed)} 个包")
    
    if failed:
        print("\n⚠️  需要安装的包:")
        for package in failed:
            print(f"   pip install {package}")
    
    # 测试基本功能
    test_basic_functionality()
    
    # 总结
    print("\n" + "=" * 60)
    if len(failed) == 0:
        print("🎉 所有P1级依赖验证通过！可以开始后续开发。")
        return 0
    else:
        print(f"⚠️  还有 {len(failed)} 个依赖需要安装，请先完成依赖安装。")
        return 1

if __name__ == "__main__":
    sys.exit(main())



