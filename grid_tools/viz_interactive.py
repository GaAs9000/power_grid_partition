"""
交互式可视化模块 - 基于plotly和pandapower
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
import logging

try:
    import pandapower as pp
    import pandapower.plotting as pplt
    HAS_PANDAPOWER = True
except ImportError:
    HAS_PANDAPOWER = False
    
from .palettes import get_palette
from .config import VizConfig

logger = logging.getLogger(__name__)


class InteractiveVisualizer:
    """交互式电网分区可视化器"""
    
    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()
        self.colors = None
        
    def plot_partition(self, 
                      env: Any,
                      mpc_data: Optional[Dict] = None,
                      title: str = "Interactive Power Grid Partition",
                      save_path: Optional[str] = None,
                      show: bool = True) -> go.Figure:
        """
        创建交互式分区可视化
        
        Args:
            env: 环境对象
            mpc_data: MATPOWER数据（可选）
            title: 标题
            save_path: 保存路径
            show: 是否显示
            
        Returns:
            plotly Figure对象
        """
        # 获取颜色
        self.colors = get_palette(env.K + 1)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Network Topology', 'Load/Gen Balance', 'Coupling Matrix',
                'Region Statistics', 'Node Distribution', 'Performance Metrics'
            ),
            specs=[
                [{'type': 'scatter', 'rowspan': 2}, {'type': 'bar'}, {'type': 'heatmap'}],
                [None, {'type': 'pie'}, {'type': 'scatter'}]
            ],
            column_widths=[0.5, 0.25, 0.25],
            row_heights=[0.6, 0.4],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # 1. 网络拓扑
        self._plot_network_topology(fig, env, row=1, col=1)
        
        # 2. 负荷/发电平衡
        self._plot_load_gen_balance(fig, env, row=1, col=2)
        
        # 3. 耦合矩阵
        self._plot_coupling_matrix(fig, env, row=1, col=3)
        
        # 4. 节点分布饼图
        self._plot_node_distribution(fig, env, row=2, col=2)
        
        # 5. 性能指标
        self._plot_performance_metrics(fig, env, row=2, col=3)
        
        # 更新布局
        fig.update_layout(
            title_text=title,
            title_font_size=20,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # 保存或显示
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")
            
        if show:
            fig.show()
            
        return fig
    
    def plot_pandapower(self,
                       env: Any,
                       mpc_data: Dict,
                       title: str = "pandapower Visualization",
                       save_path: Optional[str] = None) -> None:
        """
        使用pandapower创建专业电网图
        
        Args:
            env: 环境对象
            mpc_data: MATPOWER数据
            title: 标题  
            save_path: 保存路径
        """
        if not HAS_PANDAPOWER:
            logger.warning("pandapower not installed. Skipping pandapower visualization.")
            return
            
        # 创建pandapower网络
        net = self._create_pandapower_net(env, mpc_data)
        
        # 运行潮流计算
        try:
            pp.runpp(net, algorithm='nr', init='dc')
            logger.info("Power flow converged successfully")
        except:
            logger.warning("Power flow did not converge")
        
        # 创建可视化
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=self.config.DEFAULT_FIGSIZE)
        
        # 设置母线颜色
        bus_colors = []
        for i in range(len(net.bus)):
            region = env.z[i].item() if torch.is_tensor(env.z[i]) else env.z[i]
            bus_colors.append(self.colors[region])
        
        # 创建集合
        bus_collection = pplt.create_bus_collection(
            net, net.bus.index,
            size=80, color=bus_colors, zorder=2
        )
        
        # 线路集合
        line_colors = []
        line_widths = []
        
        for idx, line in net.line.iterrows():
            from_region = env.z[line.from_bus].item()
            to_region = env.z[line.to_bus].item()
            
            if from_region != to_region and from_region > 0 and to_region > 0:
                line_colors.append('red')
                line_widths.append(3)
            else:
                line_colors.append('gray')
                line_widths.append(1)
        
        line_collection = pplt.create_line_collection(
            net, net.line.index,
            color=line_colors,
            linewidths=line_widths,
            use_bus_geodata=True
        )
        
        # 添加到图中
        ax.add_collection(bus_collection)
        ax.add_collection(line_collection)
        
        # 添加标签和信息
        pplt.draw_collections([bus_collection, line_collection], ax=ax)
        
        ax.set_title(title, fontsize=16)
        ax.axis('equal')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def _plot_network_topology(self, fig: go.Figure, env: Any, row: int, col: int) -> None:
        """绘制网络拓扑"""
        # 构建图
        G = nx.Graph()
        edge_array = env.edge_index.cpu().numpy() if torch.is_tensor(env.edge_index) else env.edge_index
        
        for i in range(env.N):
            G.add_node(i)
            
        edge_set = set()
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            edge_set.add((min(u, v), max(u, v)))
            
        G.add_edges_from(list(edge_set))
        
        # 计算布局
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # 添加边
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # 判断是否跨区域
            z0 = env.z[edge[0]].item() if torch.is_tensor(env.z[edge[0]]) else env.z[edge[0]]
            z1 = env.z[edge[1]].item() if torch.is_tensor(env.z[edge[1]]) else env.z[edge[1]]
            
            if z0 != z1 and z0 > 0 and z1 > 0:
                color = 'red'
                width = 3
            else:
                color = 'lightgray'
                width = 1
                
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # 添加所有边到图中
        for trace in edge_trace:
            fig.add_trace(trace, row=row, col=col)
        
        # 添加节点
        for region in range(env.K + 1):
            mask = (env.z == region)
            if mask.any():
                node_indices = torch.where(mask)[0].cpu().numpy()
                
                # 节点属性
                x_coords = [pos[i][0] for i in node_indices]
                y_coords = [pos[i][1] for i in node_indices]
                
                # 节点大小
                sizes = []
                texts = []
                hovertexts = []
                
                for i in node_indices:
                    load = env.Pd_pu[i].item() if torch.is_tensor(env.Pd_pu[i]) else env.Pd_pu[i]
                    size = 10 + load * 30
                    sizes.append(size)
                    texts.append(str(i + 1))
                    
                    # 悬停文本
                    gen = env.Pg_pu[i].item() if hasattr(env, 'Pg_pu') else 0
                    hovertext = (
                        f"Bus {i+1}<br>"
                        f"Region: {region}<br>"
                        f"Load: {load:.3f} p.u.<br>"
                        f"Gen: {gen:.3f} p.u."
                    )
                    hovertexts.append(hovertext)
                
                # 添加节点轨迹
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers+text',
                        marker=dict(
                            size=sizes,
                            color=self.colors[region],
                            line=dict(color='black', width=1)
                        ),
                        text=texts,
                        textposition='center',
                        hovertext=hovertexts,
                        hoverinfo='text',
                        name=f'Region {region}' if region > 0 else 'Unassigned'
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(showgrid=False, showticklabels=False, row=row, col=col)
        fig.update_yaxes(showgrid=False, showticklabels=False, row=row, col=col)
    
    def _plot_load_gen_balance(self, fig: go.Figure, env: Any, row: int, col: int) -> None:
        """绘制负荷发电平衡图"""
        region_loads = []
        region_gens = []
        region_names = []
        
        for k in range(1, env.K + 1):
            mask = (env.z == k)
            if mask.any():
                load = env.Pd_pu[mask].sum().item()
                gen = env.Pg_pu[mask].sum().item() if hasattr(env, 'Pg_pu') else 0
                region_loads.append(load)
                region_gens.append(gen)
                region_names.append(f'R{k}')
        
        # 负荷
        fig.add_trace(
            go.Bar(
                name='Load',
                x=region_names,
                y=region_loads,
                marker_color='lightcoral',
                text=[f'{v:.2f}' for v in region_loads],
                textposition='outside'
            ),
            row=row, col=col
        )
        
        # 发电
        fig.add_trace(
            go.Bar(
                name='Generation',
                x=region_names,
                y=region_gens,
                marker_color='lightgreen',
                text=[f'{v:.2f}' for v in region_gens],
                textposition='outside'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Region", row=row, col=col)
        fig.update_yaxes(title_text="Power (p.u.)", row=row, col=col)
    
    def _plot_coupling_matrix(self, fig: go.Figure, env: Any, row: int, col: int) -> None:
        """绘制耦合矩阵热图"""
        coupling_matrix = self._compute_coupling_matrix(env)
        
        fig.add_trace(
            go.Heatmap(
                z=coupling_matrix,
                x=[f'R{i+1}' for i in range(env.K)],
                y=[f'R{i+1}' for i in range(env.K)],
                colorscale='RdBu_r',
                text=np.round(coupling_matrix, 3),
                texttemplate='%{text}',
                textfont=dict(size=10),
                showscale=True,
                colorbar=dict(title="Coupling")
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="To Region", row=row, col=col)
        fig.update_yaxes(title_text="From Region", row=row, col=col)
    
    def _plot_node_distribution(self, fig: go.Figure, env: Any, row: int, col: int) -> None:
        """绘制节点分布饼图"""
        region_counts = []
        region_labels = []
        
        for k in range(env.K + 1):
            count = (env.z == k).sum().item()
            if count > 0:
                region_counts.append(count)
                region_labels.append(f'Region {k}' if k > 0 else 'Unassigned')
        
        fig.add_trace(
            go.Pie(
                labels=region_labels,
                values=region_counts,
                marker=dict(colors=[self.colors[i] for i in range(len(region_labels))]),
                textinfo='label+percent',
                hole=0.3
            ),
            row=row, col=col
        )
    
    def _plot_performance_metrics(self, fig: go.Figure, env: Any, row: int, col: int) -> None:
        """绘制性能指标"""
        metrics = env.get_partition_metrics()
        
        # 选择关键指标
        metric_names = ['Load CV', 'Coupling', 'Lines']
        metric_values = [
            metrics.get('load_cv', 0),
            metrics.get('total_coupling', 0),
            metrics.get('inter_region_lines', 0)
        ]
        
        # 归一化处理
        max_values = [1.0, max(metric_values[1], 1), max(metric_values[2], 1)]
        normalized_values = [v/m for v, m in zip(metric_values, max_values)]
        
        fig.add_trace(
            go.Scatter(
                x=metric_names,
                y=normalized_values,
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(width=2),
                fill='tozeroy',
                name='Metrics'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Metric", row=row, col=col)
        fig.update_yaxes(title_text="Normalized Value", range=[0, 1], row=row, col=col)
    
    def _compute_coupling_matrix(self, env: Any) -> np.ndarray:
        """计算区域间耦合矩阵"""
        coupling_matrix = np.zeros((env.K, env.K))
        edge_array = env.edge_index.cpu().numpy() if torch.is_tensor(env.edge_index) else env.edge_index
        
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            region_u = env.z[u].item() if torch.is_tensor(env.z[u]) else env.z[u]
            region_v = env.z[v].item() if torch.is_tensor(env.z[v]) else env.z[v]
            
            if region_u > 0 and region_v > 0 and region_u != region_v:
                admittance = env.admittance[i].item() if hasattr(env, 'admittance') else 1.0
                coupling_matrix[region_u-1, region_v-1] += admittance
                
        return coupling_matrix
    
    def _create_pandapower_net(self, env: Any, mpc_data: Dict) -> 'pp.pandapowerNet':
        """创建pandapower网络"""
        net = pp.create_empty_network()
        
        # 添加母线
        for i, bus_data in enumerate(mpc_data['bus']):
            pp.create_bus(
                net,
                vn_kv=bus_data[9],
                name=f"Bus {i+1}",
                index=i,
                geodata=(0, 0)  # 可以后续更新
            )
        
        # 添加线路
        baseMVA = mpc_data['baseMVA']
        for branch_data in mpc_data['branch']:
            from_bus = int(branch_data[0]) - 1
            to_bus = int(branch_data[1]) - 1
            
            # 转换参数
            r_pu = branch_data[2]
            x_pu = branch_data[3]
            
            pp.create_line_from_parameters(
                net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=1,
                r_ohm_per_km=r_pu * 100,  # 简化转换
                x_ohm_per_km=x_pu * 100,
                c_nf_per_km=0,
                max_i_ka=branch_data[5] / 1000 if branch_data[5] > 0 else baseMVA / 1000
            )
        
        # 添加负荷
        for i, bus_data in enumerate(mpc_data['bus']):
            if bus_data[2] != 0:
                pp.create_load(
                    net,
                    bus=i,
                    p_mw=bus_data[2],
                    q_mvar=bus_data[3]
                )
        
        # 添加发电机
        for gen_data in mpc_data['gen']:
            bus_idx = int(gen_data[0]) - 1
            pp.create_gen(
                net,
                bus=bus_idx,
                p_mw=gen_data[1],
                vm_pu=gen_data[5]
            )
        
        # 添加外部电网（slack）
        slack_buses = np.where(mpc_data['bus'][:, 1] == 3)[0]
        if len(slack_buses) > 0:
            pp.create_ext_grid(net, bus=slack_buses[0])
        
        return net