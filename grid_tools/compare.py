"""
分区方法比较模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from .metrics import PartitionEvaluator
from .palettes import get_palette

logger = logging.getLogger(__name__)


class PartitionComparator:
    """
    比较不同分区方法的性能
    """
    
    def __init__(self, evaluator: Optional[PartitionEvaluator] = None):
        self.evaluator = evaluator or PartitionEvaluator()
        self.comparison_results = pd.DataFrame()
        self.normalized_results = pd.DataFrame()
        
    def compare_methods(self,
                       environments: Dict[str, Any],
                       save_results: bool = True) -> pd.DataFrame:
        """
        比较多种分区方法
        
        Args:
            environments: {方法名: 环境} 字典
            save_results: 是否保存结果
            
        Returns:
            比较结果DataFrame
        """
        results = []
        
        logger.info(f"Comparing {len(environments)} partition methods...")
        
        for method_name, env in environments.items():
            logger.info(f"Evaluating {method_name}...")
            
            # 评估
            metrics = self.evaluator.evaluate_partition(env)
            metrics['method'] = method_name
            metrics['timestamp'] = datetime.now().isoformat()
            
            results.append(metrics)
        
        # 创建DataFrame
        self.comparison_results = pd.DataFrame(results)
        
        # 归一化结果
        self._normalize_results()
        
        if save_results:
            self._save_results()
        
        return self.comparison_results
    
    def _normalize_results(self) -> None:
        """归一化结果用于比较"""
        df = self.comparison_results.copy()
        
        # 定义需要反转的指标（越小越好）
        inverse_metrics = [
            'load_cv', 'load_std', 'size_cv', 'total_coupling',
            'inter_region_lines', 'avg_coupling', 'max_region_coupling'
        ]
        
        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
        
        # Min-Max归一化
        normalized_df = df.copy()
        
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].values
                
                # 处理无穷值
                values[np.isinf(values)] = np.nan
                
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                
                if max_val > min_val:
                    if col in inverse_metrics:
                        # 反转：越小越好变成越大越好
                        normalized_df[col] = 1 - (values - min_val) / (max_val - min_val)
                    else:
                        normalized_df[col] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0.5
        
        self.normalized_results = normalized_df
    
    def plot_comparison_radar(self,
                            metrics_to_plot: Optional[List[str]] = None,
                            save_path: Optional[str] = None,
                            show: bool = True) -> go.Figure:
        """
        绘制雷达图比较
        
        Args:
            metrics_to_plot: 要绘制的指标列表
            save_path: 保存路径
            show: 是否显示
            
        Returns:
            plotly Figure对象
        """
        if self.normalized_results.empty:
            logger.warning("No comparison results available")
            return None
        
        if metrics_to_plot is None:
            metrics_to_plot = [
                'load_cv', 'total_coupling', 'modularity',
                'connected_region_ratio', 'n1_security_score',
                'critical_nodes_balance', 'overall_score'
            ]
        
        # 过滤存在的指标
        available_metrics = [m for m in metrics_to_plot 
                           if m in self.normalized_results.columns]
        
        if not available_metrics:
            logger.warning("No valid metrics to plot")
            return None
        
        fig = go.Figure()
        
        # 为每个方法添加轨迹
        for _, row in self.normalized_results.iterrows():
            values = [row[metric] for metric in available_metrics]
            
            # 闭合雷达图
            values.append(values[0])
            metrics_labels = available_metrics + [available_metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_labels,
                fill='toself',
                name=row['method'],
                opacity=0.6
            ))
        
        # 更新布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                )
            ),
            showlegend=True,
            title={
                'text': "Partition Methods Comparison - Radar Chart",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Radar plot saved to {save_path}")
        
        if show:
            fig.show()
        
        return fig
    
    def plot_comparison_bars(self,
                           metrics_to_plot: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制柱状图比较
        
        Args:
            metrics_to_plot: 要绘制的指标
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        if self.comparison_results.empty:
            logger.warning("No comparison results available")
            return None
        
        if metrics_to_plot is None:
            metrics_to_plot = [
                'load_cv', 'total_coupling', 'modularity',
                'inter_region_lines', 'overall_score', 'completion_rate'
            ]
        
        # 过滤存在的指标
        available_metrics = [m for m in metrics_to_plot 
                           if m in self.comparison_results.columns]
        
        n_metrics = len(available_metrics)
        if n_metrics == 0:
            logger.warning("No valid metrics to plot")
            return None
        
        # 创建子图
        fig, axes = plt.subplots(
            2, 3, 
            figsize=(15, 10),
            squeeze=False
        )
        axes = axes.flatten()
        
        # 为每个指标创建柱状图
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                ax = axes[i]
                
                # 获取数据
                methods = self.comparison_results['method'].values
                values = self.comparison_results[metric].values
                
                # 创建柱状图
                bars = ax.bar(methods, values)
                
                # 着色：最好的值用绿色
                colors = []
                if metric in ['load_cv', 'total_coupling', 'inter_region_lines']:
                    # 越小越好
                    best_idx = np.argmin(values)
                else:
                    # 越大越好
                    best_idx = np.argmax(values)
                
                for j, bar in enumerate(bars):
                    if j == best_idx:
                        bar.set_color('green')
                        bar.set_alpha(0.8)
                    else:
                        bar.set_color('lightgray')
                
                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                
                # 设置标题和标签
                ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
                ax.set_ylabel('Value', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Partition Methods Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bar plot saved to {save_path}")
        
        return fig
    
    def generate_report(self,
                       save_path: str = "partition_comparison_report.html",
                       include_plots: bool = True) -> str:
        """
        生成综合比较报告
        
        Args:
            save_path: 报告保存路径
            include_plots: 是否包含图表
            
        Returns:
            HTML报告内容
        """
        if self.comparison_results.empty:
            logger.warning("No comparison results available")
            return ""
        
        # 生成报告内容
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Power Grid Partition Comparison Report</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .best {{
            background-color: #90EE90;
            font-weight: bold;
        }}
        .summary-box {{
            background-color: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            display: inline-block;
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 15px;
            margin: 10px;
            min-width: 200px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .timestamp {{
            color: #888;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Power Grid Partition Methods Comparison Report</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            {self._generate_executive_summary()}
        </div>
        
        <h2>Overall Comparison</h2>
        {self._generate_overall_comparison_table()}
        
        <h2>Key Metrics Summary</h2>
        {self._generate_key_metrics_cards()}
        
        <h2>Detailed Results</h2>
        {self._generate_detailed_table()}
        
        <h2>Method Rankings</h2>
        {self._generate_rankings_table()}
        
        <h2>Recommendations</h2>
        {self._generate_recommendations()}
        
        <h2>Technical Details</h2>
        {self._generate_technical_details()}
    </div>
</body>
</html>
        """
        
        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {save_path}")
        
        return html_content
    
    def _generate_executive_summary(self) -> str:
        """生成执行摘要"""
        if 'overall_score' in self.comparison_results.columns:
            best_method = self.comparison_results.loc[
                self.comparison_results['overall_score'].idxmax(), 'method'
            ]
            best_score = self.comparison_results['overall_score'].max()
        else:
            best_method = "N/A"
            best_score = 0
        
        n_methods = len(self.comparison_results)
        
        summary = f"""
        <p>This report compares <strong>{n_methods}</strong> different power grid 
        partitioning methods based on comprehensive evaluation metrics.</p>
        <p>The <strong>best performing method</strong> is <span style="color: #4CAF50; 
        font-weight: bold;">{best_method}</span> with an overall score of 
        <strong>{best_score:.3f}</strong>.</p>
        """
        
        return summary
    
    def _generate_overall_comparison_table(self) -> str:
        """生成总体比较表"""
        # 选择关键指标
        key_metrics = [
            'method', 'overall_score', 'load_cv', 'total_coupling',
            'modularity', 'connected_region_ratio'
        ]
        
        available_metrics = [m for m in key_metrics 
                           if m in self.comparison_results.columns]
        
        df_display = self.comparison_results[available_metrics].copy()
        
        # 标记最佳值
        for col in available_metrics:
            if col not in ['method']:
                if col in ['load_cv', 'total_coupling']:
                    # 越小越好
                    best_idx = df_display[col].idxmin()
                else:
                    # 越大越好
                    best_idx = df_display[col].idxmax()
                
                # 添加样式标记
                df_display.loc[best_idx, col] = f'<span class="best">{df_display.loc[best_idx, col]:.4f}</span>'
        
        # 转换为HTML
        html_table = df_display.to_html(
            index=False,
            escape=False,
            float_format=lambda x: f'{x:.4f}'
        )
        
        return html_table
    
    def _generate_key_metrics_cards(self) -> str:
        """生成关键指标卡片"""
        cards_html = '<div style="text-align: center;">'
        
        # 定义要显示的关键指标
        key_metrics = [
            ('Best Overall Score', 'overall_score', True),
            ('Lowest Load CV', 'load_cv', False),
            ('Lowest Coupling', 'total_coupling', False),
            ('Best Modularity', 'modularity', True)
        ]
        
        for title, metric, maximize in key_metrics:
            if metric in self.comparison_results.columns:
                if maximize:
                    best_val = self.comparison_results[metric].max()
                    best_method = self.comparison_results.loc[
                        self.comparison_results[metric].idxmax(), 'method'
                    ]
                else:
                    best_val = self.comparison_results[metric].min()
                    best_method = self.comparison_results.loc[
                        self.comparison_results[metric].idxmin(), 'method'
                    ]
                
                cards_html += f'''
                <div class="metric-card">
                    <h4>{title}</h4>
                    <div class="metric-value">{best_val:.4f}</div>
                    <p>{best_method}</p>
                </div>
                '''
        
        cards_html += '</div>'
        
        return cards_html
    
    def _generate_detailed_table(self) -> str:
        """生成详细结果表"""
        # 选择所有数值列
        numeric_cols = self.comparison_results.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        # 添加方法列
        cols_to_show = ['method'] + numeric_cols
        
        # 创建显示DataFrame
        df_display = self.comparison_results[cols_to_show].copy()
        
        # 转换为HTML
        html_table = df_display.to_html(
            index=False,
            float_format=lambda x: f'{x:.4f}',
            classes='detailed-table'
        )
        
        return html_table
    
    def _generate_rankings_table(self) -> str:
        """生成排名表"""
        rankings = []
        
        # 对每个指标进行排名
        metrics_to_rank = [
            ('Load Balance (CV)', 'load_cv', False),
            ('Total Coupling', 'total_coupling', False),
            ('Modularity', 'modularity', True),
            ('Overall Score', 'overall_score', True)
        ]
        
        for metric_name, metric_col, ascending in metrics_to_rank:
            if metric_col in self.comparison_results.columns:
                # 排序
                sorted_df = self.comparison_results.sort_values(
                    metric_col,
                    ascending=ascending
                ).reset_index(drop=True)
                
                # 创建排名
                for i, row in sorted_df.iterrows():
                    rankings.append({
                        'Metric': metric_name,
                        'Rank': i + 1,
                        'Method': row['method'],
                        'Value': f"{row[metric_col]:.4f}"
                    })
        
        df_rankings = pd.DataFrame(rankings)
        
        if not df_rankings.empty:
            # 转换为HTML
            html_table = df_rankings.to_html(index=False)
        else:
            html_table = "<p>No ranking data available</p>"
        
        return html_table
    
    def _generate_recommendations(self) -> str:
        """生成推荐建议"""
        recommendations = []
        
        # 基于不同场景的推荐
        scenarios = [
            {
                'scenario': 'Balanced Load Distribution',
                'metric': 'load_cv',
                'minimize': True,
                'description': 'minimizing load imbalance across regions'
            },
            {
                'scenario': 'Minimal Inter-region Coupling',
                'metric': 'total_coupling',
                'minimize': True,
                'description': 'reducing electrical coupling between regions'
            },
            {
                'scenario': 'Maximum Network Modularity',
                'metric': 'modularity',
                'minimize': False,
                'description': 'maximizing natural community structure'
            },
            {
                'scenario': 'Overall Best Performance',
                'metric': 'overall_score',
                'minimize': False,
                'description': 'balanced performance across all metrics'
            }
        ]
        
        rec_html = '<ul>'
        
        for scenario in scenarios:
            if scenario['metric'] in self.comparison_results.columns:
                if scenario['minimize']:
                    best_idx = self.comparison_results[scenario['metric']].idxmin()
                else:
                    best_idx = self.comparison_results[scenario['metric']].idxmax()
                
                best_method = self.comparison_results.loc[best_idx, 'method']
                best_value = self.comparison_results.loc[best_idx, scenario['metric']]
                
                rec_html += f'''
                <li>
                    <strong>{scenario['scenario']}:</strong> 
                    Use <span style="color: #4CAF50; font-weight: bold;">{best_method}</span> 
                    for {scenario['description']} 
                    (achieves {scenario['metric']} = {best_value:.4f})
                </li>
                '''
        
        rec_html += '</ul>'
        
        # 添加通用建议
        rec_html += '''
        <h3>General Recommendations:</h3>
        <ul>
            <li>For real-time applications, consider computational efficiency alongside quality metrics</li>
            <li>Validate results with N-1 contingency analysis before deployment</li>
            <li>Consider hybrid approaches that combine strengths of different methods</li>
            <li>Monitor performance under varying load conditions</li>
        </ul>
        '''
        
        return rec_html
    
    def _generate_technical_details(self) -> str:
        """生成技术细节"""
        details = f"""
        <h3>Evaluation Metrics Used:</h3>
        <ul>
            <li><strong>Load CV:</strong> Coefficient of variation of regional loads (lower is better)</li>
            <li><strong>Total Coupling:</strong> Sum of admittances between regions (lower is better)</li>
            <li><strong>Modularity:</strong> Network modularity score (higher is better)</li>
            <li><strong>N-1 Security:</strong> Robustness under single contingencies (higher is better)</li>
            <li><strong>Overall Score:</strong> Weighted combination of all metrics (higher is better)</li>
        </ul>
        
        <h3>Evaluation Parameters:</h3>
        <ul>
            <li>Number of regions (K): {self.comparison_results.iloc[0].get('K', 'N/A')}</li>
            <li>Total nodes evaluated: {self.comparison_results.iloc[0].get('N', 'N/A')}</li>
            <li>Timestamp: {datetime.now().isoformat()}</li>
        </ul>
        """
        
        return details
    
    def _save_results(self) -> None:
        """保存结果到CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        self.comparison_results.to_csv(
            f'partition_comparison_{timestamp}.csv',
            index=False
        )
        
        # 保存归一化结果
        self.normalized_results.to_csv(
            f'partition_comparison_normalized_{timestamp}.csv',
            index=False
        )
        
        logger.info(f"Results saved with timestamp {timestamp}")