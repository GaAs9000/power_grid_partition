"""
分区质量评估模块
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import logging

from .config import MetricsConfig

logger = logging.getLogger(__name__)


class PartitionEvaluator:
    """
    电网分区质量评估器
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.metrics = {}
        self.evaluation_history = []
        
    def evaluate_partition(self, 
                         env: Any, 
                         run_power_flow: bool = None) -> Dict[str, float]:
        """
        全面评估分区质量
        
        Args:
            env: 电网环境
            run_power_flow: 是否运行潮流计算
            
        Returns:
            评估指标字典
        """
        if run_power_flow is None:
            run_power_flow = self.config.ENABLE_POWER_FLOW
            
        # 准备numpy数据（避免反复GPU-CPU同步）
        self._prepare_numpy_data(env)
        
        metrics = {}
        
        # 1. 基础指标
        logger.info("Computing basic metrics...")
        metrics.update(self._compute_basic_metrics(env))
        
        # 2. 电气指标
        logger.info("Computing electrical metrics...")
        metrics.update(self._compute_electrical_metrics(env))
        
        # 3. 图论指标
        logger.info("Computing graph metrics...")
        metrics.update(self._compute_graph_metrics(env))
        
        # 4. 鲁棒性指标
        if self.config.N1_SECURITY_CHECK:
            logger.info("Computing robustness metrics...")
            metrics.update(self._compute_robustness_metrics(env))
        
        # 5. 潮流指标（如果需要）
        if run_power_flow:
            logger.info("Computing power flow metrics...")
            metrics.update(self._compute_power_flow_metrics(env))
        
        # 6. 综合评分
        metrics['overall_score'] = self._compute_overall_score(metrics)
        
        self.metrics = metrics
        self.evaluation_history.append(metrics.copy())
        
        return metrics
    
    def _prepare_numpy_data(self, env: Any) -> None:
        """准备numpy格式数据，避免重复转换"""
        self.z_np = env.z.cpu().numpy() if torch.is_tensor(env.z) else env.z
        self.edge_index_np = (env.edge_index.cpu().numpy() 
                             if torch.is_tensor(env.edge_index) 
                             else env.edge_index)
        self.Pd_pu_np = (env.Pd_pu.cpu().numpy() 
                        if torch.is_tensor(env.Pd_pu) 
                        else env.Pd_pu)
        self.Pg_pu_np = (env.Pg_pu.cpu().numpy() 
                        if torch.is_tensor(env.Pg_pu) 
                        else env.Pg_pu) if hasattr(env, 'Pg_pu') else np.zeros_like(self.Pd_pu_np)
        
        if hasattr(env, 'admittance'):
            self.admittance_np = (env.admittance.cpu().numpy() 
                                 if torch.is_tensor(env.admittance) 
                                 else env.admittance)
        else:
            self.admittance_np = np.ones(self.edge_index_np.shape[1])
    
    def _compute_basic_metrics(self, env: Any) -> Dict[str, float]:
        """计算基础分区指标"""
        metrics = {}
        
        # 分区完成度
        assigned_nodes = np.sum(self.z_np > 0)
        metrics['completion_rate'] = assigned_nodes / env.N
        
        # 负荷均衡
        region_loads = []
        region_gens = []
        region_sizes = []
        
        for k in range(1, env.K + 1):
            mask = (self.z_np == k)
            if np.any(mask):
                load = np.sum(self.Pd_pu_np[mask])
                gen = np.sum(self.Pg_pu_np[mask])
                size = np.sum(mask)
                
                region_loads.append(load)
                region_gens.append(gen)
                region_sizes.append(size)
        
        if region_loads:
            loads = np.array(region_loads)
            gens = np.array(region_gens)
            sizes = np.array(region_sizes)
            
            # 负荷指标
            metrics['load_mean'] = np.mean(loads)
            metrics['load_std'] = np.std(loads)
            metrics['load_cv'] = np.std(loads) / np.mean(loads) if np.mean(loads) > 0 else 0
            metrics['load_gini'] = self._gini_coefficient(loads)
            metrics['load_balance_ratio'] = np.max(loads) / np.min(loads) if np.min(loads) > 0 else np.inf
            
            # 发电指标
            if np.sum(gens) > 0:
                metrics['gen_mean'] = np.mean(gens)
                metrics['gen_std'] = np.std(gens)
                metrics['gen_cv'] = np.std(gens) / np.mean(gens) if np.mean(gens) > 0 else 0
            
            # 大小指标
            metrics['size_mean'] = np.mean(sizes)
            metrics['size_std'] = np.std(sizes)
            metrics['size_cv'] = np.std(sizes) / np.mean(sizes)
            
            # 自给率
            self_sufficiency = []
            for load, gen in zip(loads, gens):
                if load > 0:
                    self_sufficiency.append(min(gen / load, 1.0))
            if self_sufficiency:
                metrics['avg_self_sufficiency'] = np.mean(self_sufficiency)
        
        return metrics
    
    def _compute_electrical_metrics(self, env: Any) -> Dict[str, float]:
        """计算电气指标"""
        metrics = {}
        
        # 总耦合强度
        total_coupling = 0.0
        inter_region_lines = 0
        coupling_by_region_pair = defaultdict(float)
        
        # 遍历所有边
        for i in range(self.edge_index_np.shape[1]):
            u, v = self.edge_index_np[0, i], self.edge_index_np[1, i]
            region_u = self.z_np[u]
            region_v = self.z_np[v]
            
            if region_u != region_v and region_u > 0 and region_v > 0:
                coupling = self.admittance_np[i]
                total_coupling += coupling
                inter_region_lines += 1
                
                # 记录区域对之间的耦合
                pair = tuple(sorted([region_u, region_v]))
                coupling_by_region_pair[pair] += coupling
        
        metrics['total_coupling'] = total_coupling
        metrics['inter_region_lines'] = inter_region_lines
        metrics['avg_coupling'] = total_coupling / inter_region_lines if inter_region_lines > 0 else 0
        
        # 计算最大区域间耦合
        if coupling_by_region_pair:
            max_coupling = max(coupling_by_region_pair.values())
            metrics['max_region_coupling'] = max_coupling
        
        # 电气模块度
        metrics['electrical_modularity'] = self._compute_electrical_modularity(env)
        
        # 边界节点比例
        boundary_nodes = self._identify_boundary_nodes(env)
        metrics['boundary_node_ratio'] = len(boundary_nodes) / env.N
        
        return metrics
    
    def _compute_graph_metrics(self, env: Any) -> Dict[str, float]:
        """计算图论指标"""
        metrics = {}
        
        # 构建NetworkX图
        G = nx.Graph()
        for i in range(env.N):
            G.add_node(i, region=self.z_np[i])
        
        # 添加边
        edge_set = set()
        for i in range(self.edge_index_np.shape[1]):
            u, v = self.edge_index_np[0, i], self.edge_index_np[1, i]
            if u != v:  # 避免自环
                edge_set.add((min(u, v), max(u, v)))
        
        for u, v in edge_set:
            # 找到对应的导纳
            idx = None
            for i in range(self.edge_index_np.shape[1]):
                if ((self.edge_index_np[0, i] == u and self.edge_index_np[1, i] == v) or
                    (self.edge_index_np[0, i] == v and self.edge_index_np[1, i] == u)):
                    idx = i
                    break
            
            weight = self.admittance_np[idx] if idx is not None else 1.0
            G.add_edge(u, v, weight=weight)
        
        # 计算模块度
        partition = defaultdict(list)
        for i in range(env.N):
            if self.z_np[i] > 0:
                partition[self.z_np[i] - 1].append(i)
        
        if partition:
            communities = list(partition.values())
            metrics['modularity'] = nx.community.modularity(G, communities, weight='weight')
        
        # 检查每个区域的连通性
        connected_regions = 0
        total_regions = 0
        
        for k in range(1, env.K + 1):
            region_nodes = [i for i in range(env.N) if self.z_np[i] == k]
            if len(region_nodes) > 1:
                total_regions += 1
                subgraph = G.subgraph(region_nodes)
                if nx.is_connected(subgraph):
                    connected_regions += 1
                    metrics[f'region_{k}_connected'] = 1.0
                else:
                    metrics[f'region_{k}_connected'] = 0.0
                    # 计算连通分量数
                    components = list(nx.connected_components(subgraph))
                    metrics[f'region_{k}_components'] = len(components)
        
        if total_regions > 0:
            metrics['connected_region_ratio'] = connected_regions / total_regions
        
        # 计算边密度
        for k in range(1, env.K + 1):
            region_nodes = [i for i in range(env.N) if self.z_np[i] == k]
            if len(region_nodes) > 1:
                subgraph = G.subgraph(region_nodes)
                density = nx.density(subgraph)
                metrics[f'region_{k}_density'] = density
        
        return metrics
    
    def _compute_robustness_metrics(self, env: Any) -> Dict[str, float]:
        """计算鲁棒性指标"""
        metrics = {}
        
        # N-1安全性评分
        n1_score = self._evaluate_n1_security(env)
        metrics['n1_security_score'] = n1_score
        
        # 关键节点分布
        critical_nodes = self._identify_critical_nodes(env)
        metrics['critical_nodes_count'] = len(critical_nodes)
        metrics['critical_nodes_ratio'] = len(critical_nodes) / env.N
        
        # 评估关键节点在各区域的分布
        critical_distribution = self._evaluate_critical_distribution(env, critical_nodes)
        metrics['critical_nodes_balance'] = critical_distribution
        
        # 冗余度评估
        redundancy_score = self._evaluate_redundancy(env)
        metrics['redundancy_score'] = redundancy_score
        
        return metrics
    
    def _compute_power_flow_metrics(self, env: Any) -> Dict[str, float]:
        """计算潮流相关指标"""
        metrics = {}
        
        try:
            # 这里需要实际的潮流计算
            # 简化示例
            metrics['power_flow_feasible'] = 1.0
            metrics['max_line_loading'] = 0.0
            metrics['max_voltage_deviation'] = 0.0
            metrics['total_losses'] = 0.0
            
        except Exception as e:
            logger.warning(f"Power flow calculation failed: {e}")
            metrics['power_flow_feasible'] = 0.0
        
        return metrics
    
    def _compute_overall_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        # 定义权重
        weights = {
            'load_cv': -0.3,  # 负向指标
            'total_coupling': -0.2,  # 负向指标
            'modularity': 0.2,  # 正向指标
            'connected_region_ratio': 0.1,  # 正向指标
            'n1_security_score': 0.1,  # 正向指标
            'critical_nodes_balance': 0.1  # 正向指标
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                # 处理无穷值
                if np.isinf(value):
                    value = 10.0 if weight < 0 else 0.0
                
                score += weight * value
        
        # 归一化到0-1
        return max(0.0, min(1.0, score + 0.5))
    
    def _gini_coefficient(self, x: np.ndarray) -> float:
        """计算基尼系数"""
        if len(x) == 0:
            return 0.0
            
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        
        if cumsum[-1] == 0:
            return 0.0
            
        return (2 * np.sum((np.arange(1, n+1) * sorted_x))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _compute_electrical_modularity(self, env: Any) -> float:
        """计算基于电气距离的模块度"""
        total_admittance = np.sum(self.admittance_np)
        if total_admittance == 0:
            return 0.0
            
        inter_region_admittance = 0.0
        
        for i in range(self.edge_index_np.shape[1]):
            u, v = self.edge_index_np[0, i], self.edge_index_np[1, i]
            if self.z_np[u] != self.z_np[v] and self.z_np[u] > 0 and self.z_np[v] > 0:
                inter_region_admittance += self.admittance_np[i]
        
        return 1.0 - (inter_region_admittance / total_admittance)
    
    def _identify_boundary_nodes(self, env: Any) -> Set[int]:
        """识别边界节点"""
        boundary = set()
        
        for i in range(self.edge_index_np.shape[1]):
            u, v = self.edge_index_np[0, i], self.edge_index_np[1, i]
            if self.z_np[u] != self.z_np[v]:
                if self.z_np[u] > 0:
                    boundary.add(u)
                if self.z_np[v] > 0:
                    boundary.add(v)
        
        return boundary
    
    def _evaluate_n1_security(self, env: Any) -> float:
        """评估N-1安全性"""
        # 简化实现：检查移除任一线路后区域是否仍连通
        # 实际应该运行N-1潮流分析
        
        G = nx.Graph()
        for i in range(env.N):
            G.add_node(i)
        
        edge_list = []
        for i in range(self.edge_index_np.shape[1]):
            u, v = self.edge_index_np[0, i], self.edge_index_np[1, i]
            if u != v:
                edge_list.append((u, v))
        
        G.add_edges_from(edge_list)
        
        # 对每个区域检查N-1连通性
        security_scores = []
        
        for k in range(1, env.K + 1):
            region_nodes = [i for i in range(env.N) if self.z_np[i] == k]
            if len(region_nodes) <= 1:
                continue
                
            subgraph = G.subgraph(region_nodes)
            if not nx.is_connected(subgraph):
                security_scores.append(0.0)
                continue
            
            # 检查移除每条边后的连通性
            region_edges = list(subgraph.edges())
            if len(region_edges) == 0:
                security_scores.append(1.0)
                continue
                
            secure_count = 0
            for edge in region_edges:
                temp_graph = subgraph.copy()
                temp_graph.remove_edge(*edge)
                if nx.is_connected(temp_graph):
                    secure_count += 1
            
            security_scores.append(secure_count / len(region_edges))
        
        return np.mean(security_scores) if security_scores else 1.0
    
    def _identify_critical_nodes(self, env: Any) -> List[int]:
        """识别关键节点"""
        critical = []
        
        # 发电机节点
        gen_threshold = 0.0
        for i in range(env.N):
            if self.Pg_pu_np[i] > gen_threshold:
                critical.append(i)
        
        # 高负荷节点
        load_mean = np.mean(self.Pd_pu_np)
        load_threshold = load_mean * self.config.CRITICAL_NODE_THRESHOLD
        
        for i in range(env.N):
            if self.Pd_pu_np[i] > load_threshold and i not in critical:
                critical.append(i)
        
        return critical
    
    def _evaluate_critical_distribution(self, env: Any, critical_nodes: List[int]) -> float:
        """评估关键节点分布均衡性"""
        if not critical_nodes:
            return 1.0
        
        region_critical_counts = []
        for k in range(1, env.K + 1):
            count = sum(1 for node in critical_nodes if self.z_np[node] == k)
            region_critical_counts.append(count)
        
        counts = np.array(region_critical_counts)
        if counts.sum() == 0:
            return 1.0
        
        # 使用熵来衡量分布均匀性
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(env.K)
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _evaluate_redundancy(self, env: Any) -> float:
        """评估系统冗余度"""
        # 简化实现：基于平均节点度
        G = nx.Graph()
        for i in range(env.N):
            G.add_node(i)
            
        for i in range(self.edge_index_np.shape[1]):
            u, v = self.edge_index_np[0, i], self.edge_index_np[1, i]
            if u != v:
                G.add_edge(u, v)
        
        avg_degree = np.mean([d for n, d in G.degree()])
        
        # 归一化到0-1
        return min(avg_degree / 4.0, 1.0)  # 假设平均度4是理想值