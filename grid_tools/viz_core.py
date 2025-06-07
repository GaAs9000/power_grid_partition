"""
核心可视化模块 - 基于matplotlib的静态图
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import matplotlib.patches as mpatches
import logging

from .palettes import get_palette
from .config import VizConfig

logger = logging.getLogger(__name__)


class Visualizer:
    """电网分区可视化器 - matplotlib版本"""
    
    def __init__(self, config: Optional[VizConfig] = None):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config or VizConfig()
        self._pos_cache = {}  # 布局缓存
        
    def plot_power_grid_partition(self, 
                                 env: Any,
                                 title: str = "Power Grid Partition", 
                                 save_path: Optional[str] = None, 
                                 show_metrics: bool = True,
                                 color_scheme: str = 'default') -> None:
        """
        绘制电网分区结果
        
        Args:
            env: 电网环境
            title: 图标题
            save_path: 保存路径
            show_metrics: 是否显示指标
            color_scheme: 配色方案
        """
        # 获取颜色
        self.colors = get_palette(env.K + 1, scheme=color_scheme)
        
        # 创建图形
        if show_metrics:
            fig = self._create_figure_with_metrics()
            ax_main = fig.add_subplot(fig.add_gridspec(2, 2, width_ratios=[3, 1], 
                                                       height_ratios=[3, 1])[:, 0])
            ax_metrics = fig.add_subplot(fig.add_gridspec(2, 2, width_ratios=[3, 1], 
                                                          height_ratios=[3, 1])[0, 1])
            ax_load = fig.add_subplot(fig.add_gridspec(2, 2, width_ratios=[3, 1], 
                                                       height_ratios=[3, 1])[1, 1])
        else:
            fig, ax_main = plt.subplots(figsize=self.config.DEFAULT_FIGSIZE)
            ax_metrics = None
            ax_load = None
        
        # 构建图
        G, edge_list = self._build_graph(env)
        
        # 计算样式
        node_colors, node_sizes = self._calc_node_styles(env)
        
        # 获取或计算布局
        pos = self._get_layout(G, env)
        
        # 绘制静态图
        self._draw_static(ax_main, G, pos, edge_list, env, node_colors, node_sizes)
        
        # 绘制指标
        if show_metrics:
            self._draw_metrics(fig, env, ax_metrics, ax_load)
        
        # 添加图例
        self._add_legend(ax_main, env)
        
        # 设置标题
        ax_main.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if not save_path:  # 只在不保存时显示
            plt.show()
        else:
            plt.close()
    
    def _create_figure_with_metrics(self) -> plt.Figure:
        """创建带指标面板的图形"""
        fig = plt.figure(figsize=(self.config.DEFAULT_FIGSIZE[0] + 4, 
                                 self.config.DEFAULT_FIGSIZE[1]))
        return fig
    
    def _build_graph(self, env: Any) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
        """
        构建NetworkX图
        
        Args:
            env: 环境对象
            
        Returns:
            G: NetworkX图
            edge_list: 边列表
        """
        G = nx.Graph()
        
        # 添加节点
        for i in range(env.N):
            G.add_node(i)
        
        # 添加边 - 使用set去重
        edge_set = set()
        edge_array = env.edge_index.cpu().numpy() if torch.is_tensor(env.edge_index) else env.edge_index
        
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            edge_set.add((min(u, v), max(u, v)))  # 确保无向边只添加一次
        
        edge_list = list(edge_set)
        G.add_edges_from(edge_list)
        
        return G, edge_list
    
    def _calc_node_styles(self, env: Any) -> Tuple[List[str], List[float]]:
        """
        计算节点样式
        
        Args:
            env: 环境对象
            
        Returns:
            node_colors: 节点颜色列表
            node_sizes: 节点大小列表
        """
        node_colors = []
        node_sizes = []
        
        # 获取最大负荷用于归一化
        max_load = env.Pd_pu.max().item() if env.Pd_pu.max() > 0 else 1.0
        
        for i in range(env.N):
            # 颜色
            region = env.z[i].item() if torch.is_tensor(env.z[i]) else env.z[i]
            node_colors.append(self.colors[region])
            
            # 大小 - 归一化处理
            load = env.Pd_pu[i].item() if torch.is_tensor(env.Pd_pu[i]) else env.Pd_pu[i]
            normalized_load = load / max_load
            size = self.config.DEFAULT_NODE_SIZE + normalized_load * (
                self.config.MAX_NODE_SIZE - self.config.DEFAULT_NODE_SIZE
            )
            node_sizes.append(size)
        
        return node_colors, node_sizes
    
    def _get_layout(self, G: nx.Graph, env: Any) -> Dict[int, Tuple[float, float]]:
        """
        获取或计算布局
        
        Args:
            G: NetworkX图
            env: 环境对象
            
        Returns:
            节点位置字典
        """
        # 生成缓存键
        cache_key = f"{env.N}_{len(G.edges)}"
        
        if cache_key in self._pos_cache:
            return self._pos_cache[cache_key]
        
        # 计算新布局
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        self._pos_cache[cache_key] = pos
        
        return pos
    
    def _draw_static(self, ax: plt.Axes, G: nx.Graph, pos: Dict, 
                    edge_list: List[Tuple[int, int]], env: Any,
                    node_colors: List[str], node_sizes: List[float]) -> None:
        """绘制静态网络图"""
        
        # 绘制所有边
        nx.draw_networkx_edges(G, pos, alpha=self.config.DEFAULT_EDGE_ALPHA, ax=ax)
        
        # 高亮跨区域边
        inter_region_edges = []
        for u, v in edge_list:
            z_u = env.z[u].item() if torch.is_tensor(env.z[u]) else env.z[u]
            z_v = env.z[v].item() if torch.is_tensor(env.z[v]) else env.z[v]
            
            if z_u != z_v and z_u > 0 and z_v > 0:
                inter_region_edges.append((u, v))
        
        if inter_region_edges:
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=inter_region_edges,
                edge_color=self.config.INTER_REGION_EDGE_COLOR,
                width=self.config.INTER_REGION_EDGE_WIDTH,
                alpha=self.config.INTER_REGION_EDGE_ALPHA,
                ax=ax
            )
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.axis('off')
    
    def _draw_metrics(self, fig: plt.Figure, env: Any, 
                     ax_metrics: plt.Axes, ax_load: plt.Axes) -> None:
        """绘制指标面板"""
        
        # 获取分区指标
        metrics = env.get_partition_metrics()
        
        # 指标表格
        ax_metrics.axis('off')
        ax_metrics.set_title('Partition Metrics', fontsize=14, fontweight='bold')
        
        # 创建指标数据
        metric_data = []
        metric_data.append(['Total Nodes', env.N])
        metric_data.append(['Assigned Nodes', (env.z > 0).sum().item()])
        metric_data.append(['Unassigned Nodes', (env.z == 0).sum().item()])
        metric_data.append(['', ''])  # 空行
        metric_data.append(['Load CV', f"{metrics.get('load_cv', 0):.4f}"])
        metric_data.append(['Total Coupling', f"{metrics.get('total_coupling', 0):.4f}"])
        metric_data.append(['Inter-region Lines', metrics.get('inter_region_lines', 0)])
        
        # 绘制表格
        table = ax_metrics.table(cellText=metric_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 负荷分布柱状图
        self._draw_load_distribution(ax_load, env)
    
    def _draw_load_distribution(self, ax: plt.Axes, env: Any) -> None:
        """绘制负荷分布图"""
        ax.set_title('Load Distribution', fontsize=12, fontweight='bold')
        
        region_loads = []
        region_gens = []
        region_labels = []
        
        for k in range(1, env.K + 1):
            mask = (env.z == k)
            if mask.any():
                load = env.Pd_pu[mask].sum().item()
                gen = env.Pg_pu[mask].sum().item() if hasattr(env, 'Pg_pu') else 0
                region_loads.append(load)
                region_gens.append(gen)
                region_labels.append(f'R{k}')
        
        if region_loads:
            x = np.arange(len(region_labels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, region_loads, width, 
                           label='Load', color='lightcoral')
            bars2 = ax.bar(x + width/2, region_gens, width,
                           label='Generation', color='lightgreen')
            
            ax.set_ylabel('Power (p.u.)', fontsize=10)
            ax.set_xlabel('Region', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(region_labels)
            ax.legend()
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _add_legend(self, ax: plt.Axes, env: Any) -> None:
        """添加图例"""
        legend_elements = []
        
        for k in range(env.K + 1):
            if k == 0:
                label = 'Unassigned'
            else:
                # 计算该区域节点数
                count = (env.z == k).sum().item()
                label = f'Region {k} ({count} nodes)'
            
            legend_elements.append(
                mpatches.Patch(color=self.colors[k], label=label)
            )
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.15, 1), frameon=True, fancybox=True)