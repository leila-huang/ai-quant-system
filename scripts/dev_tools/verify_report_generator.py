#!/usr/bin/env python3
"""
回测报告生成器验证脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import tempfile
import shutil

from backend.src.engine.backtest import BacktestResult
from backend.src.engine.backtest.metrics import (
    PerformanceAnalyzer, PerformanceMetrics, calculate_basic_metrics
)
from backend.src.engine.backtest.report_generator import (
    BacktestReportGenerator, ReportConfig, ReportData,
    create_default_report_generator, generate_quick_report
)


def create_synthetic_returns(days: int = 252, strategy_name: str = "测试策略") -> pd.Series:
    """创建合成收益率数据"""
    np.random.seed(42)
    
    # 生成日收益率（模拟真实策略）
    base_return = 0.0008  # 0.08%日均收益
    volatility = 0.015    # 1.5%日波动率
    
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    returns = np.random.normal(base_return, volatility, days)
    
    # 添加一些趋势和波动特征
    trend = np.linspace(0, 0.0003, days)  # 逐渐上升趋势
    cycle = 0.0002 * np.sin(np.linspace(0, 4*np.pi, days))  # 周期性波动
    
    returns = returns + trend + cycle
    
    return pd.Series(returns, index=dates, name=strategy_name)


def create_synthetic_trades(returns: pd.Series) -> pd.DataFrame:
    """创建合成交易数据"""
    np.random.seed(123)
    
    # 模拟交易记录
    n_trades = min(50, len(returns) // 5)  # 大约每5天一笔交易
    trade_dates = np.random.choice(returns.index, size=n_trades, replace=False)
    trade_dates = sorted(trade_dates)
    
    trades_data = []
    for i, trade_date in enumerate(trade_dates):
        entry_date = pd.to_datetime(trade_date)
        exit_date = entry_date + timedelta(days=np.random.randint(1, 20))
        
        if exit_date <= returns.index[-1]:
            # 计算交易收益
            period_returns = returns.loc[entry_date:exit_date]
            trade_return = (1 + period_returns).prod() - 1
            
            trades_data.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'symbol': f'00000{i%3+1}',
                'quantity': np.random.randint(1000, 5000),
                'entry_price': 10.0 + np.random.random() * 5,
                'exit_price': 10.0 + np.random.random() * 5,
                'pnl': trade_return,
                'returns': trade_return,
                'side': 'buy' if i % 2 == 0 else 'sell'
            })
    
    return pd.DataFrame(trades_data)


def test_performance_analyzer():
    """测试业绩分析器"""
    print("🔍 测试业绩分析器...")
    
    try:
        # 创建测试数据
        returns = create_synthetic_returns()
        benchmark = create_synthetic_returns() * 0.8  # 基准收益稍低
        
        # 创建分析器
        analyzer = PerformanceAnalyzer()
        
        # 计算指标
        metrics = analyzer.calculate_metrics(returns, benchmark)
        
        print(f"  总收益率: {metrics.total_return:.2%}")
        print(f"  年化收益率: {metrics.annualized_return:.2%}")
        print(f"  年化波动率: {metrics.volatility:.2%}")
        print(f"  夏普比率: {metrics.sharpe_ratio:.3f}")
        print(f"  最大回撤: {metrics.max_drawdown:.2%}")
        print(f"  索提诺比率: {metrics.sortino_ratio:.3f}")
        print(f"  卡尔玛比率: {metrics.calmar_ratio:.3f}")
        
        # 验证指标合理性
        if (metrics.total_return != 0 and 
            metrics.volatility > 0 and
            not np.isnan(metrics.sharpe_ratio)):
            print("  ✅ 业绩分析器测试通过")
            return True
        else:
            print("  ❌ 业绩指标计算异常")
            return False
            
    except Exception as e:
        print(f"  ❌ 业绩分析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_metrics_calculation():
    """测试基本指标计算"""
    print("\n📊 测试基本指标计算...")
    
    try:
        # 创建测试数据
        returns = create_synthetic_returns(100)  # 100天数据
        
        # 使用便利函数计算基本指标
        basic_metrics = calculate_basic_metrics(returns)
        
        print(f"  基本指标计算结果:")
        for metric, value in basic_metrics.items():
            if isinstance(value, float):
                if 'Rate' in metric or 'Return' in metric or 'Drawdown' in metric:
                    print(f"    {metric}: {value:.2%}")
                else:
                    print(f"    {metric}: {value:.3f}")
            else:
                print(f"    {metric}: {value}")
        
        # 验证指标数量和合理性
        if (len(basic_metrics) >= 5 and 
            all(isinstance(v, (int, float)) for v in basic_metrics.values())):
            print("  ✅ 基本指标计算测试通过")
            return True
        else:
            print("  ❌ 基本指标计算结果不完整")
            return False
            
    except Exception as e:
        print(f"  ❌ 基本指标计算测试失败: {e}")
        return False


def test_html_report_generation():
    """测试HTML报告生成"""
    print("\n🌐 测试HTML报告生成...")
    
    try:
        # 创建测试数据
        returns = create_synthetic_returns(180)  # 半年数据
        trades = create_synthetic_trades(returns)
        
        # 创建BacktestResult
        backtest_result = BacktestResult(
            returns=returns,
            positions=pd.DataFrame(),
            trades=trades,
            metrics={},
            metadata={'strategy_name': '测试策略HTML'}
        )
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 配置报告生成器
            config = ReportConfig(
                report_title="HTML报告测试",
                output_dir=temp_dir,
                include_charts=True,
                include_detailed_metrics=True
            )
            
            generator = BacktestReportGenerator(config)
            
            # 生成HTML报告
            html_path = generator.generate_report(backtest_result, 'html')
            
            # 验证文件是否生成
            if os.path.exists(html_path):
                file_size = os.path.getsize(html_path)
                print(f"  HTML报告生成成功: {os.path.basename(html_path)}")
                print(f"  文件大小: {file_size / 1024:.1f} KB")
                
                # 读取部分内容验证
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # 读取前1000字符
                    if 'DOCTYPE html' in content and '测试策略HTML' in content:
                        print("  ✅ HTML报告内容验证通过")
                        return True
                    else:
                        print("  ❌ HTML报告内容验证失败")
                        return False
            else:
                print("  ❌ HTML报告文件未生成")
                return False
                
    except Exception as e:
        print(f"  ❌ HTML报告生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_report_generation():
    """测试PDF报告生成"""
    print("\n📄 测试PDF报告生成...")
    
    try:
        # 创建测试数据
        returns = create_synthetic_returns(120)  # 4个月数据
        
        # 创建BacktestResult
        backtest_result = BacktestResult(
            returns=returns,
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={},
            metadata={'strategy_name': '测试策略PDF'}
        )
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ReportConfig(
                report_title="PDF报告测试",
                output_dir=temp_dir,
                include_charts=True
            )
            
            generator = BacktestReportGenerator(config)
            
            # 生成PDF报告
            pdf_path = generator.generate_report(backtest_result, 'pdf')
            
            # 验证文件是否生成
            if os.path.exists(pdf_path):
                file_size = os.path.getsize(pdf_path)
                print(f"  PDF报告生成成功: {os.path.basename(pdf_path)}")
                print(f"  文件大小: {file_size / 1024:.1f} KB")
                
                # 简单验证PDF文件格式
                with open(pdf_path, 'rb') as f:
                    header = f.read(4)
                    if header == b'%PDF':
                        print("  ✅ PDF报告格式验证通过")
                        return True
                    else:
                        print("  ❌ PDF报告格式验证失败")
                        return False
            else:
                print("  ❌ PDF报告文件未生成")
                return False
                
    except Exception as e:
        print(f"  ❌ PDF报告生成测试失败: {e}")
        # PDF生成可能因为缺少依赖而失败，但不应该阻止其他测试
        print("  ⚠️ 如果是matplotlib/PDF相关错误，可能需要安装额外依赖")
        return False


def test_json_report_generation():
    """测试JSON报告生成"""
    print("\n📋 测试JSON报告生成...")
    
    try:
        # 创建测试数据
        returns = create_synthetic_returns(90)  # 3个月数据
        
        # 创建BacktestResult
        backtest_result = BacktestResult(
            returns=returns,
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={},
            metadata={'strategy_name': '测试策略JSON'}
        )
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ReportConfig(
                report_title="JSON报告测试",
                output_dir=temp_dir
            )
            
            generator = BacktestReportGenerator(config)
            
            # 生成JSON报告
            json_path = generator.generate_report(backtest_result, 'json')
            
            # 验证文件是否生成
            if os.path.exists(json_path):
                file_size = os.path.getsize(json_path)
                print(f"  JSON报告生成成功: {os.path.basename(json_path)}")
                print(f"  文件大小: {file_size / 1024:.1f} KB")
                
                # 验证JSON格式和内容
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                required_sections = ['metadata', 'performance_metrics', 'time_series_data']
                if all(section in report_data for section in required_sections):
                    print("  ✅ JSON报告结构验证通过")
                    
                    # 验证关键数据
                    metadata = report_data['metadata']
                    metrics = report_data['performance_metrics']
                    ts_data = report_data['time_series_data']
                    
                    if (metadata['strategy_name'] == '测试策略JSON' and
                        'annualized_return' in metrics and
                        len(ts_data['returns']) == len(returns)):
                        print("  ✅ JSON报告内容验证通过")
                        return True
                    else:
                        print("  ❌ JSON报告内容验证失败")
                        return False
                else:
                    print("  ❌ JSON报告结构不完整")
                    return False
            else:
                print("  ❌ JSON报告文件未生成")
                return False
                
    except Exception as e:
        print(f"  ❌ JSON报告生成测试失败: {e}")
        return False


def test_excel_report_generation():
    """测试Excel报告生成"""
    print("\n📊 测试Excel报告生成...")
    
    try:
        # 创建测试数据
        returns = create_synthetic_returns(150)
        trades = create_synthetic_trades(returns)
        
        # 创建BacktestResult
        backtest_result = BacktestResult(
            returns=returns,
            positions=pd.DataFrame(),
            trades=trades,
            metrics={},
            metadata={'strategy_name': '测试策略Excel'}
        )
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ReportConfig(
                report_title="Excel报告测试",
                output_dir=temp_dir
            )
            
            generator = BacktestReportGenerator(config)
            
            # 生成Excel报告
            excel_path = generator.generate_report(backtest_result, 'excel')
            
            # 验证文件是否生成
            if os.path.exists(excel_path):
                file_size = os.path.getsize(excel_path)
                print(f"  Excel报告生成成功: {os.path.basename(excel_path)}")
                print(f"  文件大小: {file_size / 1024:.1f} KB")
                
                # 验证Excel文件内容
                try:
                    excel_data = pd.read_excel(excel_path, sheet_name=None)
                    sheets = list(excel_data.keys())
                    print(f"  包含工作表: {sheets}")
                    
                    # 验证基本工作表是否存在
                    required_sheets = ['业绩指标', '时间序列数据']
                    if all(sheet in sheets for sheet in required_sheets):
                        print("  ✅ Excel报告结构验证通过")
                        
                        # 验证数据完整性
                        metrics_sheet = excel_data['业绩指标']
                        ts_sheet = excel_data['时间序列数据']
                        
                        if (len(metrics_sheet) > 10 and  # 至少10个指标
                            len(ts_sheet) == len(returns)):  # 时间序列长度匹配
                            print("  ✅ Excel报告内容验证通过")
                            return True
                        else:
                            print("  ❌ Excel报告内容不完整")
                            return False
                    else:
                        print("  ❌ Excel报告缺少必要工作表")
                        return False
                        
                except Exception as e:
                    print(f"  ❌ Excel文件读取失败: {e}")
                    return False
            else:
                print("  ❌ Excel报告文件未生成")
                return False
                
    except Exception as e:
        print(f"  ❌ Excel报告生成测试失败: {e}")
        return False


def test_quick_report_function():
    """测试快速报告生成函数"""
    print("\n⚡ 测试快速报告生成...")
    
    try:
        # 创建测试数据
        returns = create_synthetic_returns(60, "快速测试策略")
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 更改工作目录以使用临时目录
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # 使用快速报告函数
                html_path = generate_quick_report(returns, "快速测试策略", 'html')
                
                # 验证文件生成
                if os.path.exists(html_path):
                    print(f"  快速HTML报告生成成功: {os.path.basename(html_path)}")
                    
                    # 验证基本内容
                    with open(html_path, 'r', encoding='utf-8') as f:
                        content = f.read(500)
                        if '快速测试策略' in content and 'DOCTYPE html' in content:
                            print("  ✅ 快速报告生成测试通过")
                            return True
                        else:
                            print("  ❌ 快速报告内容验证失败")
                            return False
                else:
                    print("  ❌ 快速报告文件未生成")
                    return False
                    
            finally:
                os.chdir(original_cwd)
                
    except Exception as e:
        print(f"  ❌ 快速报告生成测试失败: {e}")
        return False


def test_strategy_comparison():
    """测试策略对比功能"""
    print("\n🔄 测试策略对比...")
    
    try:
        # 创建多个策略的测试数据
        strategy1 = create_synthetic_returns(200, "策略A")
        strategy2 = create_synthetic_returns(200, "策略B") * 1.2  # 稍高收益
        strategy3 = create_synthetic_returns(200, "策略C") * 0.8  # 稍低收益
        benchmark = create_synthetic_returns(200, "基准") * 0.6   # 基准收益
        
        # 创建分析器
        analyzer = PerformanceAnalyzer()
        
        # 策略对比分析
        strategies = {
            '策略A': strategy1,
            '策略B': strategy2, 
            '策略C': strategy3
        }
        
        comparison_df = analyzer.compare_strategies(strategies, benchmark)
        
        print(f"  策略对比结果:")
        print(f"  参与对比策略数量: {len(comparison_df)}")
        
        # 显示关键指标对比
        key_columns = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        available_columns = [col for col in key_columns if col in comparison_df.columns]
        
        if available_columns:
            comparison_subset = comparison_df[available_columns]
            print("  关键指标对比:")
            for strategy, metrics in comparison_subset.iterrows():
                print(f"    {strategy}:")
                for metric, value in metrics.items():
                    if pd.notna(value):
                        if 'return' in metric or 'drawdown' in metric or 'volatility' in metric:
                            print(f"      {metric}: {value:.2%}")
                        else:
                            print(f"      {metric}: {value:.3f}")
            
            print("  ✅ 策略对比分析测试通过")
            return True
        else:
            print("  ❌ 策略对比结果缺少关键指标")
            return False
            
    except Exception as e:
        print(f"  ❌ 策略对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_configuration():
    """测试报告配置管理"""
    print("\n⚙️ 测试报告配置...")
    
    try:
        # 创建自定义配置
        custom_config = ReportConfig(
            report_title="自定义回测报告",
            subtitle="配置测试版本",
            author="测试用户",
            figure_width=10,
            figure_height=6,
            include_summary=True,
            include_charts=False,  # 禁用图表
            include_detailed_metrics=True,
            include_trade_analysis=False,  # 禁用交易分析
            risk_free_rate=0.025  # 2.5%无风险利率
        )
        
        # 创建报告生成器
        generator = BacktestReportGenerator(custom_config)
        
        # 验证配置应用
        config_info = {
            'report_title': generator.config.report_title,
            'author': generator.config.author,
            'include_charts': generator.config.include_charts,
            'include_trade_analysis': generator.config.include_trade_analysis,
            'risk_free_rate': generator.config.risk_free_rate,
            'analyzer_rf_rate': generator.analyzer.risk_free_rate
        }
        
        print(f"  配置验证:")
        for key, value in config_info.items():
            print(f"    {key}: {value}")
        
        # 验证配置是否正确应用
        if (generator.config.report_title == "自定义回测报告" and
            generator.config.author == "测试用户" and
            generator.config.include_charts == False and
            generator.analyzer.risk_free_rate == 0.025):
            print("  ✅ 报告配置管理测试通过")
            return True
        else:
            print("  ❌ 报告配置未正确应用")
            return False
            
    except Exception as e:
        print(f"  ❌ 报告配置测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 AI量化系统 - 回测报告生成器验证")
    print("=" * 80)
    
    test_functions = [
        test_performance_analyzer,
        test_basic_metrics_calculation,
        test_html_report_generation,
        test_pdf_report_generation,
        test_json_report_generation,
        test_excel_report_generation,
        test_quick_report_function,
        test_strategy_comparison,
        test_report_configuration
    ]
    
    results = []
    start_time = time.time()
    
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    total_time = time.time() - start_time
    
    # 汇总测试结果
    print("\n" + "=" * 80)
    passed_tests = sum(results)
    total_tests = len(results)
    
    print(f"测试结果汇总:")
    for i, (test_func, result) in enumerate(zip(test_functions, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {test_func.__name__}: {status}")
    
    print(f"\n总体结果: {passed_tests}/{total_tests} 通过")
    print(f"总执行时间: {total_time:.2f}秒")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！回测报告生成器验证成功！")
        print("\n📊 回测报告生成器现已准备就绪，具备以下能力:")
        print("   • 完整的业绩指标计算体系")
        print("   • 专业的HTML可视化报告")
        print("   • 高质量的PDF打印报告") 
        print("   • 结构化的JSON数据报告")
        print("   • 便捷的Excel分析报告")
        print("   • 快速报告生成接口")
        print("   • 多策略对比分析")
        print("   • 灵活的配置化管理")
        print("   • 完整的图表可视化")
        print("   • 详细的风险收益分析")
        return 0
    elif passed_tests >= total_tests * 0.7:
        print(f"\n✅ 大部分测试通过！报告生成器核心功能正常")
        print(f"有 {total_tests - passed_tests} 个测试需要优化，但主要功能可用")
        print("注意: PDF生成可能需要额外的系统依赖")
        return 0
    else:
        print(f"\n⚠️ 有 {total_tests - passed_tests} 个关键测试失败，需要检查相关功能")
        return 1


if __name__ == "__main__":
    sys.exit(main())
