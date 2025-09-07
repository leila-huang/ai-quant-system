"""
回测报告生成器

生成专业的量化投资回测报告，包括完整的风险收益指标计算、
图表生成和多格式报告输出功能。
"""

import logging
import os
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import warnings
import json
import base64
from io import BytesIO
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats

from backend.src.engine.backtest import ReportGenerator, BacktestResult
from backend.src.engine.backtest.metrics import PerformanceAnalyzer, PerformanceMetrics

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """报告格式"""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    EXCEL = "excel"


class ChartType(Enum):
    """图表类型"""
    NET_VALUE = "net_value"          # 净值曲线
    DRAWDOWN = "drawdown"            # 回撤图
    RETURN_DISTRIBUTION = "return_dist"  # 收益分布
    ROLLING_METRICS = "rolling"      # 滚动指标
    CORRELATION = "correlation"      # 相关性矩阵
    RISK_RETURN = "risk_return"      # 风险收益散点图


@dataclass
class ReportConfig:
    """报告配置"""
    # 基础配置
    report_title: str = "量化回测报告"
    subtitle: str = ""
    author: str = "AI量化系统"
    
    # 图表配置
    figure_width: int = 12
    figure_height: int = 8
    dpi: int = 300
    chart_style: str = "seaborn"
    
    # 内容配置
    include_summary: bool = True
    include_charts: bool = True
    include_detailed_metrics: bool = True
    include_trade_analysis: bool = True
    include_risk_analysis: bool = True
    
    # 输出配置
    output_dir: str = "reports"
    filename_prefix: str = "backtest_report"
    
    # 基准配置
    benchmark_name: str = "基准"
    risk_free_rate: float = 0.03


@dataclass
class ReportData:
    """报告数据"""
    strategy_name: str
    returns: pd.Series
    positions: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    benchmark: Optional[pd.Series] = None
    metadata: Optional[Dict[str, Any]] = None


class BacktestReportGenerator(ReportGenerator):
    """
    回测报告生成器
    
    生成专业的量化投资回测报告，包括完整的风险收益指标计算、
    图表生成和多格式报告输出功能。
    """
    
    def __init__(self, config: ReportConfig = None):
        """
        初始化报告生成器
        
        Args:
            config: 报告配置
        """
        self.config = config or ReportConfig()
        self.analyzer = PerformanceAnalyzer(risk_free_rate=self.config.risk_free_rate)
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 图表缓存
        self.chart_cache = {}
        
        logger.info("回测报告生成器初始化完成")
    
    def generate_report(self, 
                       backtest_result: BacktestResult, 
                       output_format: str = 'html') -> str:
        """
        生成回测报告
        
        Args:
            backtest_result: 回测结果
            output_format: 输出格式
            
        Returns:
            str: 报告文件路径或内容
        """
        try:
            # 准备报告数据
            report_data = self._prepare_report_data(backtest_result)
            
            # 生成报告内容
            if output_format.lower() == 'html':
                return self._generate_html_report(report_data)
            elif output_format.lower() == 'pdf':
                return self._generate_pdf_report(report_data)
            elif output_format.lower() == 'json':
                return self._generate_json_report(report_data)
            elif output_format.lower() == 'excel':
                return self._generate_excel_report(report_data)
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")
                
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            raise
    
    def _prepare_report_data(self, backtest_result: BacktestResult) -> ReportData:
        """准备报告数据"""
        try:
            # 从BacktestResult提取数据
            returns = backtest_result.returns
            positions = backtest_result.positions
            trades = backtest_result.trades
            metadata = backtest_result.metadata or {}
            
            # 构造报告数据
            report_data = ReportData(
                strategy_name=metadata.get('strategy_name', '未命名策略'),
                returns=returns,
                positions=positions,
                trades=trades,
                metadata=metadata
            )
            
            return report_data
            
        except Exception as e:
            logger.error(f"准备报告数据失败: {e}")
            raise
    
    def _generate_html_report(self, report_data: ReportData) -> str:
        """生成HTML报告"""
        try:
            # 计算业绩指标
            metrics = self.analyzer.calculate_metrics(report_data.returns, report_data.benchmark)
            
            # 生成图表
            charts_html = self._generate_charts_html(report_data, metrics)
            
            # 构建HTML内容
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.report_title}</title>
    <style>
        body {{
            font-family: 'Arial', 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .section {{
            margin: 40px 0;
        }}
        .section h2 {{
            color: #3498db;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 10px;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header_html(report_data)}
        {self._generate_summary_html(metrics)}
        {self._generate_detailed_metrics_html(metrics)}
        {charts_html}
        {self._generate_trade_analysis_html(report_data)}
        {self._generate_footer_html()}
    </div>
</body>
</html>
            """
            
            # 保存HTML文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.filename_prefix}_{timestamp}.html"
            filepath = os.path.join(self.config.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML报告生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
            raise
    
    def _generate_header_html(self, report_data: ReportData) -> str:
        """生成报告头部HTML"""
        return f"""
        <div class="header">
            <h1>{self.config.report_title}</h1>
            <h2>{report_data.strategy_name}</h2>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>报告作者: {self.config.author}</p>
        </div>
        """
    
    def _generate_summary_html(self, metrics: PerformanceMetrics) -> str:
        """生成摘要HTML"""
        if not self.config.include_summary:
            return ""
        
        return f"""
        <div class="section">
            <h2>📊 业绩摘要</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">总收益率</div>
                    <div class="metric-value {'positive' if metrics.total_return > 0 else 'negative'}">
                        {metrics.total_return:.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">年化收益率</div>
                    <div class="metric-value {'positive' if metrics.annualized_return > 0 else 'negative'}">
                        {metrics.annualized_return:.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">年化波动率</div>
                    <div class="metric-value">{metrics.volatility:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">夏普比率</div>
                    <div class="metric-value {'positive' if metrics.sharpe_ratio > 0 else 'negative'}">
                        {metrics.sharpe_ratio:.3f}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大回撤</div>
                    <div class="metric-value negative">{metrics.max_drawdown:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">卡尔玛比率</div>
                    <div class="metric-value {'positive' if metrics.calmar_ratio > 0 else 'negative'}">
                        {metrics.calmar_ratio:.3f}
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_detailed_metrics_html(self, metrics: PerformanceMetrics) -> str:
        """生成详细指标HTML"""
        if not self.config.include_detailed_metrics:
            return ""
        
        metrics_data = [
            ("收益指标", [
                ("总收益率", f"{metrics.total_return:.2%}"),
                ("年化收益率", f"{metrics.annualized_return:.2%}"),
                ("累计收益率", f"{metrics.cumulative_return:.2%}"),
            ]),
            ("风险指标", [
                ("年化波动率", f"{metrics.volatility:.2%}"),
                ("下行波动率", f"{metrics.downside_volatility:.2%}"),
                ("最大回撤", f"{metrics.max_drawdown:.2%}"),
                ("平均回撤", f"{metrics.avg_drawdown:.2%}"),
            ]),
            ("风险调整收益", [
                ("夏普比率", f"{metrics.sharpe_ratio:.3f}"),
                ("索提诺比率", f"{metrics.sortino_ratio:.3f}"),
                ("卡尔玛比率", f"{metrics.calmar_ratio:.3f}"),
            ]),
            ("分布特征", [
                ("偏度", f"{metrics.skewness:.3f}"),
                ("峰度", f"{metrics.kurtosis:.3f}"),
                ("95% VaR", f"{metrics.var_95:.2%}"),
                ("95% CVaR", f"{metrics.cvar_95:.2%}"),
            ])
        ]
        
        tables_html = ""
        for category, items in metrics_data:
            table_rows = "".join([
                f"<tr><td>{label}</td><td>{value}</td></tr>"
                for label, value in items
            ])
            tables_html += f"""
            <h3>{category}</h3>
            <table>
                <thead>
                    <tr><th>指标</th><th>数值</th></tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            """
        
        return f"""
        <div class="section">
            <h2>📈 详细指标</h2>
            {tables_html}
        </div>
        """
    
    def _generate_charts_html(self, report_data: ReportData, metrics: PerformanceMetrics) -> str:
        """生成图表HTML"""
        if not self.config.include_charts:
            return ""
        
        charts_html = '<div class="section"><h2>📊 可视化分析</h2>'
        
        # 生成净值曲线图
        net_value_chart = self._create_net_value_chart(report_data.returns)
        if net_value_chart:
            charts_html += f'<div class="chart-container">{net_value_chart}</div>'
        
        # 生成回撤图
        drawdown_chart = self._create_drawdown_chart(report_data.returns)
        if drawdown_chart:
            charts_html += f'<div class="chart-container">{drawdown_chart}</div>'
        
        # 生成收益分布图
        distribution_chart = self._create_return_distribution_chart(report_data.returns)
        if distribution_chart:
            charts_html += f'<div class="chart-container">{distribution_chart}</div>'
        
        charts_html += '</div>'
        return charts_html
    
    def _create_net_value_chart(self, returns: pd.Series) -> str:
        """创建净值曲线图"""
        try:
            # 计算累计净值
            cumulative_returns = (1 + returns).cumprod()
            
            # 创建Plotly图表
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='策略净值',
                line=dict(color='#3498db', width=2)
            ))
            
            fig.update_layout(
                title='净值曲线',
                xaxis_title='日期',
                yaxis_title='累计净值',
                width=800,
                height=400,
                showlegend=True
            )
            
            # 转换为HTML
            return fig.to_html(include_plotlyjs='cdn', div_id="net_value_chart")
            
        except Exception as e:
            logger.warning(f"创建净值曲线图失败: {e}")
            return ""
    
    def _create_drawdown_chart(self, returns: pd.Series) -> str:
        """创建回撤图"""
        try:
            # 计算回撤序列
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            # 创建Plotly图表
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name='回撤',
                fill='tozeroy',
                line=dict(color='#e74c3c', width=1),
                fillcolor='rgba(231, 76, 60, 0.3)'
            ))
            
            fig.update_layout(
                title='回撤分析',
                xaxis_title='日期',
                yaxis_title='回撤 (%)',
                width=800,
                height=400,
                showlegend=True
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id="drawdown_chart")
            
        except Exception as e:
            logger.warning(f"创建回撤图失败: {e}")
            return ""
    
    def _create_return_distribution_chart(self, returns: pd.Series) -> str:
        """创建收益分布图"""
        try:
            # 创建直方图
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                name='收益分布',
                marker_color='#3498db',
                opacity=0.7
            ))
            
            fig.update_layout(
                title='日收益率分布',
                xaxis_title='日收益率 (%)',
                yaxis_title='频次',
                width=800,
                height=400,
                showlegend=True
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id="distribution_chart")
            
        except Exception as e:
            logger.warning(f"创建收益分布图失败: {e}")
            return ""
    
    def _generate_trade_analysis_html(self, report_data: ReportData) -> str:
        """生成交易分析HTML"""
        if not self.config.include_trade_analysis or report_data.trades is None:
            return ""
        
        try:
            trades = report_data.trades
            if trades.empty:
                return ""
            
            # 基本交易统计
            total_trades = len(trades)
            if 'pnl' in trades.columns or 'returns' in trades.columns:
                pnl_col = 'pnl' if 'pnl' in trades.columns else 'returns'
                profitable_trades = len(trades[trades[pnl_col] > 0])
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                avg_profit = trades[trades[pnl_col] > 0][pnl_col].mean() if profitable_trades > 0 else 0
                avg_loss = trades[trades[pnl_col] < 0][pnl_col].mean() if (total_trades - profitable_trades) > 0 else 0
                profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            else:
                win_rate = 0
                avg_profit = 0
                avg_loss = 0
                profit_loss_ratio = 0
            
            return f"""
            <div class="section">
                <h2>📈 交易分析</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">总交易次数</div>
                        <div class="metric-value">{total_trades}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">胜率</div>
                        <div class="metric-value">{win_rate:.2%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">盈亏比</div>
                        <div class="metric-value">{profit_loss_ratio:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">平均盈利</div>
                        <div class="metric-value positive">{avg_profit:.4f}</div>
                    </div>
                </div>
            </div>
            """
            
        except Exception as e:
            logger.warning(f"生成交易分析失败: {e}")
            return ""
    
    def _generate_footer_html(self) -> str:
        """生成报告尾部HTML"""
        return f"""
        <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d;">
            <p>报告由 {self.config.author} 自动生成</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
    
    def _generate_pdf_report(self, report_data: ReportData) -> str:
        """生成PDF报告"""
        try:
            # 计算指标
            metrics = self.analyzer.calculate_metrics(report_data.returns, report_data.benchmark)
            
            # 创建PDF文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.filename_prefix}_{timestamp}.pdf"
            filepath = os.path.join(self.config.output_dir, filename)
            
            with PdfPages(filepath) as pdf:
                # 第一页：摘要
                self._create_summary_page(pdf, report_data, metrics)
                
                # 第二页：净值曲线
                self._create_net_value_page(pdf, report_data.returns)
                
                # 第三页：回撤分析
                self._create_drawdown_page(pdf, report_data.returns)
                
                # 第四页：收益分布
                self._create_distribution_page(pdf, report_data.returns)
            
            logger.info(f"PDF报告生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成PDF报告失败: {e}")
            raise
    
    def _create_summary_page(self, pdf: PdfPages, report_data: ReportData, metrics: PerformanceMetrics):
        """创建摘要页面"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{self.config.report_title} - {report_data.strategy_name}', fontsize=16, fontweight='bold')
        
        # 关键指标表格
        ax1.axis('tight')
        ax1.axis('off')
        ax1.set_title('关键业绩指标', fontsize=14)
        
        key_metrics = [
            ['指标', '数值'],
            ['总收益率', f'{metrics.total_return:.2%}'],
            ['年化收益率', f'{metrics.annualized_return:.2%}'],
            ['年化波动率', f'{metrics.volatility:.2%}'],
            ['夏普比率', f'{metrics.sharpe_ratio:.3f}'],
            ['最大回撤', f'{metrics.max_drawdown:.2%}'],
            ['卡尔玛比率', f'{metrics.calmar_ratio:.3f}']
        ]
        
        table = ax1.table(cellText=key_metrics[1:], colLabels=key_metrics[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 风险指标
        ax2.axis('tight')
        ax2.axis('off')
        ax2.set_title('风险分析指标', fontsize=14)
        
        risk_metrics = [
            ['指标', '数值'],
            ['下行波动率', f'{metrics.downside_volatility:.2%}'],
            ['索提诺比率', f'{metrics.sortino_ratio:.3f}'],
            ['VaR (95%)', f'{metrics.var_95:.2%}'],
            ['CVaR (95%)', f'{metrics.cvar_95:.2%}'],
            ['偏度', f'{metrics.skewness:.3f}'],
            ['峰度', f'{metrics.kurtosis:.3f}']
        ]
        
        table2 = ax2.table(cellText=risk_metrics[1:], colLabels=risk_metrics[0],
                          cellLoc='center', loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 1.5)
        
        # 交易统计（如果有交易数据）
        ax3.axis('tight')
        ax3.axis('off')
        ax3.set_title('交易统计', fontsize=14)
        
        trading_stats = [
            ['指标', '数值'],
            ['胜率', f'{metrics.win_rate:.2%}'],
            ['盈亏比', f'{metrics.profit_loss_ratio:.2f}'],
            ['期望收益', f'{metrics.expectancy:.4f}']
        ]
        
        table3 = ax3.table(cellText=trading_stats[1:], colLabels=trading_stats[0],
                          cellLoc='center', loc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1, 1.5)
        
        # 其他信息
        ax4.axis('off')
        info_text = f"""
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
报告作者: {self.config.author}
数据期间: {report_data.returns.index[0].strftime('%Y-%m-%d')} 至 {report_data.returns.index[-1].strftime('%Y-%m-%d')}
样本数量: {len(report_data.returns)} 个交易日
        """
        ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_net_value_page(self, pdf: PdfPages, returns: pd.Series):
        """创建净值曲线页面"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算累计净值
        cumulative_returns = (1 + returns).cumprod()
        
        ax.plot(cumulative_returns.index, cumulative_returns.values, 'b-', linewidth=2, label='策略净值')
        ax.set_title('净值曲线', fontsize=16, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('累计净值')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_drawdown_page(self, pdf: PdfPages, returns: pd.Series):
        """创建回撤分析页面"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 计算回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # 回撤时序图
        ax1.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.3, color='red', label='回撤')
        ax1.plot(drawdown.index, drawdown.values * 100, 'r-', linewidth=1)
        ax1.set_title('回撤时序图', fontsize=14, fontweight='bold')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('回撤 (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 回撤分布直方图
        ax2.hist(drawdown.values * 100, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('回撤分布直方图', fontsize=14, fontweight='bold')
        ax2.set_xlabel('回撤 (%)')
        ax2.set_ylabel('频次')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_distribution_page(self, pdf: PdfPages, returns: pd.Series):
        """创建收益分布页面"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('收益率分布分析', fontsize=16, fontweight='bold')
        
        returns_pct = returns * 100
        
        # 直方图
        ax1.hist(returns_pct, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('日收益率直方图')
        ax1.set_xlabel('日收益率 (%)')
        ax1.set_ylabel('频次')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q图
        stats.probplot(returns_pct, dist="norm", plot=ax2)
        ax2.set_title('Q-Q图 (正态分布)')
        ax2.grid(True, alpha=0.3)
        
        # 箱线图
        ax3.boxplot(returns_pct, vert=True)
        ax3.set_title('收益率箱线图')
        ax3.set_ylabel('日收益率 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 月度收益热力图
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        years = monthly_returns.index.year.unique()
        months = monthly_returns.index.month.unique()
        
        if len(years) > 1:
            heatmap_data = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax4)
            ax4.set_title('月度收益率热力图 (%)')
            ax4.set_xlabel('月份')
            ax4.set_ylabel('年份')
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, '数据期间不足以生成\n月度收益热力图', ha='center', va='center')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _generate_json_report(self, report_data: ReportData) -> str:
        """生成JSON报告"""
        try:
            # 计算指标
            metrics = self.analyzer.calculate_metrics(report_data.returns, report_data.benchmark)
            
            # 构建报告数据
            report_dict = {
                'metadata': {
                    'report_title': self.config.report_title,
                    'strategy_name': report_data.strategy_name,
                    'generated_at': datetime.now().isoformat(),
                    'author': self.config.author,
                    'data_period': {
                        'start_date': report_data.returns.index[0].isoformat(),
                        'end_date': report_data.returns.index[-1].isoformat(),
                        'total_days': len(report_data.returns)
                    }
                },
                'performance_metrics': metrics.to_dict(),
                'time_series_data': {
                    'dates': report_data.returns.index.strftime('%Y-%m-%d').tolist(),
                    'returns': report_data.returns.tolist(),
                    'cumulative_returns': (1 + report_data.returns).cumprod().tolist()
                }
            }
            
            # 保存JSON文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.filename_prefix}_{timestamp}.json"
            filepath = os.path.join(self.config.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON报告生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成JSON报告失败: {e}")
            raise
    
    def _generate_excel_report(self, report_data: ReportData) -> str:
        """生成Excel报告"""
        try:
            # 计算指标
            metrics = self.analyzer.calculate_metrics(report_data.returns, report_data.benchmark)
            
            # 创建Excel文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.filename_prefix}_{timestamp}.xlsx"
            filepath = os.path.join(self.config.output_dir, filename)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 业绩指标表
                metrics_df = pd.DataFrame.from_dict(metrics.to_dict(), orient='index', columns=['数值'])
                metrics_df.to_excel(writer, sheet_name='业绩指标')
                
                # 时间序列数据
                ts_data = pd.DataFrame({
                    '日期': report_data.returns.index,
                    '日收益率': report_data.returns.values,
                    '累计净值': (1 + report_data.returns).cumprod().values
                })
                ts_data.to_excel(writer, sheet_name='时间序列数据', index=False)
                
                # 如果有交易数据，添加交易明细
                if report_data.trades is not None and not report_data.trades.empty:
                    report_data.trades.to_excel(writer, sheet_name='交易明细', index=False)
                
                # 如果有持仓数据，添加持仓明细
                if report_data.positions is not None and not report_data.positions.empty:
                    report_data.positions.to_excel(writer, sheet_name='持仓明细', index=False)
            
            logger.info(f"Excel报告生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成Excel报告失败: {e}")
            raise
    
    def generate_comparison_report(self, 
                                  strategies: Dict[str, pd.Series],
                                  benchmark: Optional[pd.Series] = None,
                                  output_format: str = 'html') -> str:
        """
        生成策略对比报告
        
        Args:
            strategies: 策略收益率字典
            benchmark: 基准收益率
            output_format: 输出格式
            
        Returns:
            str: 报告文件路径
        """
        try:
            # 计算所有策略的指标
            comparison_results = self.analyzer.compare_strategies(strategies, benchmark)
            
            if output_format.lower() == 'html':
                return self._generate_comparison_html(comparison_results, strategies, benchmark)
            elif output_format.lower() == 'excel':
                return self._generate_comparison_excel(comparison_results, strategies, benchmark)
            else:
                raise ValueError(f"策略对比报告不支持格式: {output_format}")
                
        except Exception as e:
            logger.error(f"生成策略对比报告失败: {e}")
            raise
    
    def _generate_comparison_html(self, 
                                comparison_results: pd.DataFrame,
                                strategies: Dict[str, pd.Series],
                                benchmark: Optional[pd.Series] = None) -> str:
        """生成策略对比HTML报告"""
        # 实现策略对比HTML报告生成逻辑
        pass
    
    def _generate_comparison_excel(self, 
                                 comparison_results: pd.DataFrame,
                                 strategies: Dict[str, pd.Series],
                                 benchmark: Optional[pd.Series] = None) -> str:
        """生成策略对比Excel报告"""
        # 实现策略对比Excel报告生成逻辑
        pass


def create_default_report_generator() -> BacktestReportGenerator:
    """创建默认报告生成器"""
    config = ReportConfig(
        report_title="量化投资回测报告",
        author="AI量化系统",
        include_charts=True,
        include_detailed_metrics=True,
        include_trade_analysis=True
    )
    
    return BacktestReportGenerator(config)


def generate_quick_report(returns: pd.Series, 
                        strategy_name: str = "策略",
                        output_format: str = 'html') -> str:
    """
    快速生成报告的便利函数
    
    Args:
        returns: 收益率序列
        strategy_name: 策略名称
        output_format: 输出格式
        
    Returns:
        str: 报告文件路径
    """
    # 构造简单的BacktestResult
    backtest_result = BacktestResult(
        returns=returns,
        positions=pd.DataFrame(),
        trades=pd.DataFrame(),
        metrics={},
        metadata={'strategy_name': strategy_name}
    )
    
    # 生成报告
    generator = create_default_report_generator()
    return generator.generate_report(backtest_result, output_format)
