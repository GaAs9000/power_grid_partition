"""
强化学习环境模块
实现电网分区的MDP环境，包含物理约束和
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from collections import defaultdict, deque
import copy


class PowerGridPartitioningEnv:
    """
    电网分区环境，实现MDP定义，包含物理约束和Curriculum Learning
    """
    
    def __init__(self, 
                 node_features: torch.Tensor,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 node_embeddings: torch.Tensor,
                 K: int = 3,
                 baseMVA: float = 100.0,
                 balance_weight: float = 1.0,
                 decoupling_weight: float = 1.0,
                 physics_weight: float = 0.5,
                 connectivity_weight: float = 0.3,
                 progress_weight: float = 0.1,
                 enable_physics_constraints: bool = True,
                 device: str = 'cpu'):
        """
        初始化电网分区环境
        
        Args:
            node_features: 节点特征 [N, F]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, D]
            node_embeddings: 节点嵌入 [N, embedding_dim]
            K: 目标分区数
            baseMVA: 基准功率
            balance_weight: 均衡性奖励权重
            decoupling_weight: 解耦性奖励权重
            physics_weight: 物理约束奖励权重
            connectivity_weight: 连通性奖励权重
            progress_weight: 进度奖励权重
            enable_physics_constraints: 是否启用物理约束
            device: 计算设备
        """
        self.device = torch.device(device)
        self.node_features = node_features.to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.edge_attr = edge_attr.to(self.device)
        self.node_embeddings = node_embeddings.to(self.device)
        self.K = K
        self.baseMVA = baseMVA
        self.N = node_features.shape[0]
        self.embedding_dim = node_embeddings.shape[1]
        
        # 奖励权重
        self.w_balance = balance_weight
        self.w_decoupling = decoupling_weight
        self.w_physics = physics_weight
        self.w_connectivity = connectivity_weight
        self.w_progress = progress_weight
        self.enable_physics_constraints = enable_physics_constraints
        
        # 提取节点信息
        self.Pd_pu = node_features[:, 3]  # 有功负荷
        self.Qd_pu = node_features[:, 4]  # 无功负荷
        self.Pg_pu = node_features[:, 9]  # 有功发电
        self.Qg_pu = node_features[:, 10]  # 无功发电
        self.is_gen = node_features[:, 8]  # 是否为发电节点
        
        # 构建邻接表和图结构
        self.adj_list = self._build_adjacency_list()
        self.adj_matrix = self._build_adjacency_matrix()
        
        # 计算电气参数
        r = edge_attr[:, 0]
        x = edge_attr[:, 1]
        self.admittance = 1.0 / torch.sqrt(r**2 + x**2)
        self.impedance = torch.sqrt(r**2 + x**2)
        
        # 势能函数历史（用于reward shaping）
        self.potential_history = []
        
        # 记录历史信息
        self.history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'partitions': [],
            'metrics': []
        }
        
        # 安全层参数
        self.min_region_size = max(1, self.N // (self.K * 2))  # 最小区域大小
        self.max_region_size = self.N // self.K * 2  # 最大区域大小
        
        self.reset()
    
    def _build_adjacency_list(self) -> Dict[int, List[int]]:
        """构建邻接表"""
        adj_list = defaultdict(list)
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[:, i]
            adj_list[u.item()].append(v.item())
        return dict(adj_list)
    
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """构建邻接矩阵"""
        adj_matrix = torch.zeros(self.N, self.N, device=self.device)
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[:, i]
            adj_matrix[u, v] = 1
        return adj_matrix
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境到初始状态"""
        # 节点分配标签
        self.z = torch.zeros(self.N, dtype=torch.long, device=self.device)
        
        # 智能种子选择：选择彼此距离较远的节点
        seed_nodes = self._select_diverse_seeds()
        for k, node in enumerate(seed_nodes):
            self.z[node] = k + 1
        
        # 初始化区域嵌入
        self.region_embeddings = torch.zeros(self.K, self.embedding_dim, device=self.device)
        self._update_region_embeddings()
        
        # 初始化边界节点
        self.boundary_nodes = self._get_boundary_nodes()
        
        # 初始化未分配节点
        self.unassigned_nodes = (self.z == 0).nonzero().squeeze(-1)
        
        # 初始化全局上下文
        self.global_context = self._compute_global_context()
        
        # 重置势能历史
        self.potential_history = []
        
        # 重置历史
        self.history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'partitions': [],
            'metrics': []
        }
        
        self.t = 0
        
        return self._get_state()
    
    def _select_diverse_seeds(self) -> torch.Tensor:
        """选择多样化的种子节点（彼此距离较远）"""
        if self.K >= self.N:
            return torch.randperm(self.N)[:self.K]
        
        # 使用最短路径距离选择种子
        seeds = []
        
        # 选择第一个种子（优先选择高负荷节点）
        load_scores = self.Pd_pu + self.Qd_pu.abs() * 0.5
        first_seed = torch.argmax(load_scores).item()
        seeds.append(first_seed)
        
        # 选择剩余种子（最大化到已选种子的最小距离）
        for _ in range(1, self.K):
            distances = torch.full((self.N,), float('inf'), device=self.device)
            
            for seed in seeds:
                # 简化：使用BFS计算距离
                dist = self._bfs_distance(seed)
                distances = torch.minimum(distances, dist)
            
            # 排除已选种子
            for seed in seeds:
                distances[seed] = -1
            
            next_seed = torch.argmax(distances).item()
            seeds.append(next_seed)
        
        return torch.tensor(seeds, dtype=torch.long, device=self.device)
    
    def _bfs_distance(self, start: int) -> torch.Tensor:
        """计算从起始节点到所有节点的BFS距离"""
        distances = torch.full((self.N,), float('inf'), device=self.device)
        distances[start] = 0
        
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in self.adj_list:
                for neighbor in self.adj_list[node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
        
        return distances
    
    def _update_region_embeddings(self):
        """更新区域聚合嵌入（考虑物理特性）"""
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            if mask.any():
                # 基础嵌入：节点嵌入的加权平均
                node_weights = self.Pd_pu[mask].abs() + 1e-6  # 以负荷为权重
                node_weights = node_weights / node_weights.sum()
                weighted_embeddings = self.node_embeddings[mask] * node_weights.unsqueeze(-1)
                base_embedding = weighted_embeddings.sum(dim=0)
                
                # 添加区域统计信息
                region_load = self.Pd_pu[mask].sum()
                region_gen = self.Pg_pu[mask].sum()
                region_size = mask.sum().float() / self.N
                
                # 组合成最终嵌入
                stats = torch.tensor([region_load, region_gen, region_size], 
                                   device=self.device)
                
                # 投影统计信息到嵌入空间
                stats_embedding = torch.zeros(self.embedding_dim, device=self.device)
                stats_embedding[:3] = stats
                
                self.region_embeddings[k-1] = base_embedding * 0.8 + stats_embedding * 0.2
    
    def _get_boundary_nodes(self) -> torch.Tensor:
        """获取边界节点集合（优化版）"""
        boundary_set = set()
        
        # 使用向量化操作找边界节点
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[:, i]
            u_val, v_val = u.item(), v.item()
            
            # 未分配节点邻接已分配节点
            if self.z[u_val] == 0 and self.z[v_val] > 0:
                boundary_set.add(u_val)
            elif self.z[v_val] == 0 and self.z[u_val] > 0:
                boundary_set.add(v_val)
        
        return torch.tensor(list(boundary_set), dtype=torch.long, device=self.device) \
               if boundary_set else torch.tensor([], dtype=torch.long, device=self.device)
    
    def _compute_global_context(self) -> torch.Tensor:
        """计算增强的全局上下文向量"""
        assigned_mask = (self.z > 0)
        
        # 基础上下文
        if assigned_mask.any():
            assigned_embeddings = self.node_embeddings[assigned_mask]
            context = assigned_embeddings.mean(dim=0)
        else:
            context = torch.zeros(self.embedding_dim, device=self.device)
        
        # 分区进展
        progress = assigned_mask.float().mean()
        
        # 区域统计
        region_stats = []
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            count = mask.sum().float() / self.N
            load = self.Pd_pu[mask].sum() if mask.any() else 0.0
            region_stats.extend([count, load.item()])
        
        region_stats_tensor = torch.tensor(region_stats, device=self.device)
        
        # 平衡度指标
        if assigned_mask.sum() > 0:
            balance_metric = self._compute_balance_metric()
            coupling_metric = self._compute_total_coupling()
        else:
            balance_metric = torch.tensor(0.0, device=self.device)
            coupling_metric = torch.tensor(0.0, device=self.device)
        
        # 组合所有信息
        context = torch.cat([
            context, 
            progress.unsqueeze(0), 
            region_stats_tensor,
            balance_metric.unsqueeze(0),
            coupling_metric.unsqueeze(0)
        ])
        
        return context
    
    def _get_state(self) -> Dict[str, torch.Tensor]:
        """获取当前状态"""
        return {
            'node_embeddings': self.node_embeddings,
            'z': self.z,
            'region_embeddings': self.region_embeddings,
            'global_context': self.global_context,
            'boundary_nodes': self.boundary_nodes,
            'unassigned_nodes': self.unassigned_nodes,
            't': self.t
        }
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """获取当前有效的动作列表（带安全检查）"""
        valid_actions = []
        
        for node in self.boundary_nodes:
            valid_regions = set()
            if node.item() in self.adj_list:
                for neighbor in self.adj_list[node.item()]:
                    if self.z[neighbor] > 0:
                        region = self.z[neighbor].item()
                        # 安全检查：区域大小限制
                        if self._is_safe_assignment(node.item(), region):
                            valid_regions.add(region)
            
            for region in valid_regions:
                valid_actions.append((node.item(), region))
        
        return valid_actions
    
    def _is_safe_assignment(self, node: int, region: int) -> bool:
        """检查分配是否满足安全约束"""
        # 检查区域大小约束
        current_size = (self.z == region).sum().item()
        if current_size >= self.max_region_size:
            return False
        
        # 检查连通性（简化版）
        # TODO: 实现完整的连通性检查
        
        # 检查功率平衡约束
        if self.enable_physics_constraints:
            return self._check_power_balance_constraint(node, region)
        
        return True
    
    def _check_power_balance_constraint(self, node: int, region: int) -> bool:
        """检查功率平衡约束"""
        # 计算分配后的区域功率
        mask = (self.z == region)
        future_mask = mask.clone()
        future_mask[node] = True
        
        future_p_gen = self.Pg_pu[future_mask].sum()
        future_p_load = self.Pd_pu[future_mask].sum()
        
        # 允许一定的不平衡（考虑区域间功率交换）
        imbalance = abs(future_p_gen - future_p_load)
        max_allowed_imbalance = 0.3 * max(future_p_gen, future_p_load) + 0.1
        
        return imbalance <= max_allowed_imbalance
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作（增强版）
        
        Args:
            action: (节点索引, 区域标签)
        
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        node_idx, region = action
        
        # 记录动作前的状态
        prev_state = self._get_state()
        self.history['states'].append(prev_state)
        self.history['actions'].append(action)
        self.history['partitions'].append(self.z.clone())
        
        # 记录动作前的指标
        prev_metrics = self.get_partition_metrics()
        
        # 分配节点到区域
        self.z[node_idx] = region
        
        # 更新状态
        self._update_region_embeddings()
        self.boundary_nodes = self._get_boundary_nodes()
        self.unassigned_nodes = (self.z == 0).nonzero().squeeze(-1)
        self.global_context = self._compute_global_context()
        
        # 计算综合奖励
        reward_dict = self._compute_comprehensive_reward(action, prev_state, prev_metrics)
        reward = reward_dict['total']
        
        self.history['rewards'].append(reward)
        self.history['metrics'].append(self.get_partition_metrics())
        
        # 检查是否结束
        done = (self.z == 0).sum() == 0
        
        # 如果完成，添加完成奖励
        if done:
            completion_bonus = self._compute_completion_bonus()
            reward += completion_bonus
            reward_dict['completion_bonus'] = completion_bonus
        
        # 额外信息
        info = {
            **reward_dict,
            'num_assigned': (self.z > 0).sum().item(),
            'num_unassigned': (self.z == 0).sum().item(),
            'metrics': self.get_partition_metrics()
        }
        
        self.t += 1
        
        return self._get_state(), reward, done, info
    
    def _compute_comprehensive_reward(self, action: Tuple[int, int], 
                                    prev_state: Dict, 
                                    prev_metrics: Dict) -> Dict[str, float]:
        """计算综合奖励（包含所有改进）"""
        node_idx, region = action
        reward_dict = {}
        
        # 1. 即时奖励
        balance_reward = self._compute_balance_reward()
        decoupling_reward = self._compute_decoupling_reward(node_idx, region)
        
        reward_dict['balance'] = balance_reward * self.w_balance
        reward_dict['decoupling'] = decoupling_reward * self.w_decoupling
        
        # 2. 物理约束奖励
        if self.enable_physics_constraints:
            physics_reward = self._compute_physics_constraints_reward()
            reward_dict['physics'] = physics_reward * self.w_physics
        else:
            reward_dict['physics'] = 0.0
        
        # 3. 连通性奖励
        connectivity_reward = self._compute_connectivity_reward(node_idx, region)
        reward_dict['connectivity'] = connectivity_reward * self.w_connectivity
        
        # 4. 进步奖励（Reward Shaping）
        progress_reward = self._compute_progress_reward(prev_metrics)
        reward_dict['progress'] = progress_reward * self.w_progress
        
        # 5. 势能差奖励
        potential_reward = self._compute_potential_based_reward()
        reward_dict['potential'] = potential_reward
        
        # 总奖励
        reward_dict['total'] = sum(reward_dict.values())
        
        return reward_dict
    
    def _compute_balance_reward(self) -> float:
        """计算负荷均衡奖励（改进版）"""
        region_loads = []
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            if mask.any():
                p_load = self.Pd_pu[mask].sum()
                region_loads.append(p_load)
            else:
                region_loads.append(0.0)
        
        if len(region_loads) > 0 and max(region_loads) > 0:
            loads_tensor = torch.tensor(region_loads, device=self.device)
            
            # 使用基尼系数而不是变异系数
            gini = self._compute_gini_coefficient(loads_tensor)
            reward = -gini  # 基尼系数越小越好
        else:
            reward = 0.0
        
        return reward
    
    def _compute_gini_coefficient(self, values: torch.Tensor) -> float:
        """计算基尼系数"""
        if len(values) == 0:
            return 0.0
        
        sorted_values = torch.sort(values)[0]
        n = len(values)
        index = torch.arange(1, n + 1, device=self.device)
        
        gini = (2 * index * sorted_values).sum() / (n * sorted_values.sum()) - (n + 1) / n
        return gini.item()
    
    def _compute_decoupling_reward(self, node_idx: int, region: int) -> float:
        """计算解耦性奖励（考虑功率流）"""
        coupling_strength = 0.0
        
        if node_idx in self.adj_list:
            for neighbor in self.adj_list[node_idx]:
                if self.z[neighbor] > 0 and self.z[neighbor] != region:
                    # 找到连接边
                    edge_mask = ((self.edge_index[0] == node_idx) & (self.edge_index[1] == neighbor)) | \
                                ((self.edge_index[0] == neighbor) & (self.edge_index[1] == node_idx))
                    if edge_mask.any():
                        edge_idx = edge_mask.nonzero()[0]
                        
                        # 考虑导纳和功率流
                        admittance = self.admittance[edge_idx].item()
                        
                        # 估计功率流（简化）
                        power_diff = abs(self.Pd_pu[node_idx] - self.Pd_pu[neighbor])
                        
                        coupling_strength += admittance * (1 + power_diff)
        
        return -coupling_strength
    
    def _compute_physics_constraints_reward(self) -> float:
        """计算物理约束奖励"""
        total_penalty = 0.0
        
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            if mask.any():
                # 功率平衡约束
                P_gen = self.Pg_pu[mask].sum()
                P_load = self.Pd_pu[mask].sum()
                Q_gen = self.Qg_pu[mask].sum()
                Q_load = self.Qd_pu[mask].sum()
                
                # 有功功率不平衡
                p_imbalance = abs(P_gen - P_load) / (P_load + 1e-6)
                
                # 无功功率不平衡
                q_imbalance = abs(Q_gen - Q_load) / (abs(Q_load) + 1e-6)
                
                # 使用barrier function
                if p_imbalance > 0.8:  # 接近违反约束
                    total_penalty -= torch.log(1 - p_imbalance + 0.2)
                else:
                    total_penalty -= p_imbalance
                
                if q_imbalance > 0.8:
                    total_penalty -= 0.5 * torch.log(1 - q_imbalance + 0.2)
                else:
                    total_penalty -= 0.5 * q_imbalance
        
        return total_penalty.item() if isinstance(total_penalty, torch.Tensor) else total_penalty
    
    def _compute_connectivity_reward(self, node_idx: int, region: int) -> float:
        """计算连通性奖励"""
        # 简化版：检查节点是否有多个邻居在同一区域
        same_region_neighbors = 0
        total_neighbors = 0
        
        if node_idx in self.adj_list:
            for neighbor in self.adj_list[node_idx]:
                total_neighbors += 1
                if self.z[neighbor] == region:
                    same_region_neighbors += 1
        
        if total_neighbors > 0:
            connectivity_score = same_region_neighbors / total_neighbors
            return connectivity_score
        
        return 0.0
    
    def _compute_progress_reward(self, prev_metrics: Dict) -> float:
        """计算进步奖励"""
        curr_metrics = self.get_partition_metrics()
        
        # 负荷均衡改进
        prev_cv = prev_metrics.get('load_cv', 1.0)
        curr_cv = curr_metrics.get('load_cv', 1.0)
        balance_improvement = max(0, prev_cv - curr_cv)
        
        # 耦合度改进
        prev_coupling = prev_metrics.get('total_coupling', 0.0)
        curr_coupling = curr_metrics.get('total_coupling', 0.0)
        coupling_improvement = max(0, prev_coupling - curr_coupling) / (prev_coupling + 1e-6)
        
        return balance_improvement + coupling_improvement * 0.5
    
    def _compute_potential_based_reward(self) -> float:
        """基于势能的奖励塑造"""
        # 计算当前势能
        unassigned = (self.z == 0).sum().float()
        imbalance = self._compute_balance_metric()
        coupling = self._compute_total_coupling()
        
        potential = -0.1 * unassigned - 0.5 * imbalance - 0.3 * coupling
        
        # 计算势能差
        if self.potential_history:
            prev_potential = self.potential_history[-1]
            reward = (potential - prev_potential).item()
        else:
            reward = 0.0
        
        self.potential_history.append(potential)
        
        return reward
    
    def _compute_balance_metric(self) -> torch.Tensor:
        """计算平衡度指标"""
        region_loads = []
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            if mask.any():
                p_load = self.Pd_pu[mask].sum()
                region_loads.append(p_load)
        
        if region_loads:
            loads_tensor = torch.tensor(region_loads, device=self.device)
            if loads_tensor.mean() > 0:
                cv = loads_tensor.std() / loads_tensor.mean()
                return cv
        
        return torch.tensor(0.0, device=self.device)
    
    def _compute_total_coupling(self) -> torch.Tensor:
        """计算总耦合度"""
        total_coupling = torch.tensor(0.0, device=self.device)
        
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[:, i]
            if self.z[u] != self.z[v] and self.z[u] > 0 and self.z[v] > 0:
                total_coupling += self.admittance[i]
        
        return total_coupling
    
    def _compute_completion_bonus(self) -> float:
        """计算完成奖励"""
        metrics = self.get_partition_metrics()
        
        # 基于最终质量的奖励
        load_cv = metrics.get('load_cv', 1.0)
        coupling = metrics.get('total_coupling', 1.0)
        
        quality_score = 1.0 / (1.0 + load_cv) + 1.0 / (1.0 + coupling)
        
        return quality_score
    
    def get_partition_metrics(self) -> Dict[str, float]:
        """获取当前分区的评估指标（增强版）"""
        metrics = {}
        
        # 负荷均衡度
        region_loads = []
        region_gens = []
        region_sizes = []
        
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            if mask.any():
                p_load = self.Pd_pu[mask].sum().item()
                p_gen = self.Pg_pu[mask].sum().item()
                size = mask.sum().item()
                
                region_loads.append(p_load)
                region_gens.append(p_gen)
                region_sizes.append(size)
        
        if region_loads:
            loads_array = np.array(region_loads)
            metrics['load_mean'] = loads_array.mean()
            metrics['load_std'] = loads_array.std()
            metrics['load_cv'] = loads_array.std() / loads_array.mean() if loads_array.mean() > 0 else 0
            metrics['load_gini'] = self._compute_gini_coefficient(
                torch.tensor(region_loads, device=self.device)
            )
        
        # 功率平衡度
        if region_loads:
            imbalances = []
            for i in range(len(region_loads)):
                imbalance = abs(region_gens[i] - region_loads[i])
                imbalances.append(imbalance)
            metrics['avg_power_imbalance'] = np.mean(imbalances)
            metrics['max_power_imbalance'] = np.max(imbalances)
        
        # 总耦合强度
        total_coupling = 0.0
        inter_region_edges = 0
        
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[:, i]
            if self.z[u] != self.z[v] and self.z[u] > 0 and self.z[v] > 0:
                total_coupling += self.admittance[i].item()
                inter_region_edges += 1
        
        metrics['total_coupling'] = total_coupling
        metrics['inter_region_edges'] = inter_region_edges
        
        # 每个区域的节点数
        for k in range(1, self.K + 1):
            metrics[f'region_{k}_nodes'] = (self.z == k).sum().item()
            metrics[f'region_{k}_load'] = region_loads[k-1] if k-1 < len(region_loads) else 0
            metrics[f'region_{k}_gen'] = region_gens[k-1] if k-1 < len(region_gens) else 0
        
        # 连通性指标
        metrics['avg_connectivity'] = self._compute_avg_connectivity()
        
        return metrics
    
    def _compute_avg_connectivity(self) -> float:
        """计算平均连通性"""
        connectivity_scores = []
        
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            if mask.sum() > 1:  # 至少有2个节点
                # 计算区域内的边数
                region_nodes = mask.nonzero().squeeze(-1)
                internal_edges = 0
                
                for i in range(self.edge_index.shape[1]):
                    u, v = self.edge_index[:, i]
                    if mask[u] and mask[v]:
                        internal_edges += 1
                
                # 计算连通性分数
                n = mask.sum().item()
                max_edges = n * (n - 1) / 2
                connectivity = internal_edges / max_edges if max_edges > 0 else 0
                connectivity_scores.append(connectivity)
        
        return np.mean(connectivity_scores) if connectivity_scores else 0.0
    
    def render(self, mode: str = 'human'):
        """渲染环境状态（增强版）"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step {self.t} | Assigned: {(self.z > 0).sum().item()}/{self.N} | "
                  f"Boundary: {len(self.boundary_nodes)}")
            print(f"{'-'*60}")
            
            metrics = self.get_partition_metrics()
            
            # 打印每个区域的详细信息
            for k in range(1, self.K + 1):
                mask = (self.z == k)
                if mask.any():
                    nodes = mask.sum().item()
                    load = metrics[f'region_{k}_load']
                    gen = metrics[f'region_{k}_gen']
                    imbalance = abs(gen - load)
                    
                    print(f"Region {k}: {nodes:3d} nodes | "
                          f"Load: {load:6.3f} | Gen: {gen:6.3f} | "
                          f"Imbal: {imbalance:6.3f}")
            
            print(f"{'-'*60}")
            print(f"Load CV: {metrics['load_cv']:.4f} | "
                  f"Total Coupling: {metrics['total_coupling']:.4f} | "
                  f"Avg Connectivity: {metrics['avg_connectivity']:.4f}")
            print(f"{'='*60}")


class CurriculumEnvironment:
    """
    课程学习环境，渐进式增加难度
    """
    
    def __init__(self, base_env: PowerGridPartitioningEnv):
        """
        初始化课程学习环境
        
        Args:
            base_env: 基础环境
        """
        self.base_env = base_env
        self.difficulty_level = 0.0  # 0-1之间
        self.success_history = deque(maxlen=100)
        self.episode_count = 0
        
        # 难度参数
        self.min_preset_ratio = 0.5  # 最简单时预设50%节点
        self.max_constraint_tightness = 2.0  # 最难时约束加倍
        
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境（根据难度调整）"""
        state = self.base_env.reset()
        
        # 根据难度调整初始状态
        if self.difficulty_level < 0.3:
            # 简单：预分配部分节点
            self._preset_easy_nodes()
        elif self.difficulty_level > 0.7:
            # 困难：添加额外约束
            self._add_constraints()
        
        return state
    
    def _preset_easy_nodes(self):
        """预分配容易的节点"""
        preset_ratio = self.min_preset_ratio * (1 - self.difficulty_level / 0.3)
        num_preset = int(self.base_env.N * preset_ratio)
        
        # 为每个区域预分配一些明显的节点
        for k in range(1, self.base_env.K + 1):
            # 找到种子节点的邻居
            seed_mask = (self.base_env.z == k)
            if seed_mask.any():
                seed_nodes = seed_mask.nonzero().squeeze(-1)
                
                for seed in seed_nodes:
                    if seed.item() in self.base_env.adj_list:
                        neighbors = self.base_env.adj_list[seed.item()]
                        
                        # 分配部分邻居
                        for neighbor in neighbors[:num_preset // self.base_env.K]:
                            if self.base_env.z[neighbor] == 0:
                                self.base_env.z[neighbor] = k
        
        # 更新环境状态
        self.base_env._update_region_embeddings()
        self.base_env.boundary_nodes = self.base_env._get_boundary_nodes()
        self.base_env.unassigned_nodes = (self.base_env.z == 0).nonzero().squeeze(-1)
        self.base_env.global_context = self.base_env._compute_global_context()
    
    def _add_constraints(self):
        """添加额外约束"""
        # 收紧区域大小约束
        tightness = 1 + (self.difficulty_level - 0.7) / 0.3 * (self.max_constraint_tightness - 1)
        
        self.base_env.min_region_size = int(self.base_env.min_region_size * tightness)
        self.base_env.max_region_size = int(self.base_env.max_region_size / tightness)
        
        # 增加物理约束权重
        self.base_env.w_physics *= tightness
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool, Dict]:
        """执行动作"""
        next_state, reward, done, info = self.base_env.step(action)
        
        # 记录成功信息
        if done:
            metrics = info['metrics']
            success = self._evaluate_success(metrics)
            self.success_history.append(success)
            self.episode_count += 1
            
            # 更新难度
            if self.episode_count % 10 == 0:
                self._update_difficulty()
        
        return next_state, reward, done, info
    
    def _evaluate_success(self, metrics: Dict) -> bool:
        """评估是否成功完成任务"""
        # 定义成功标准
        load_cv_threshold = 0.3
        coupling_threshold = 2.0
        
        load_cv = metrics.get('load_cv', float('inf'))
        coupling = metrics.get('total_coupling', float('inf'))
        
        return load_cv < load_cv_threshold and coupling < coupling_threshold
    
    def _update_difficulty(self):
        """更新难度等级"""
        if len(self.success_history) < 50:
            return
        
        success_rate = np.mean(self.success_history)
        
        # 根据成功率调整难度
        if success_rate > 0.8:
            self.difficulty_level = min(1.0, self.difficulty_level + 0.1)
        elif success_rate < 0.3:
            self.difficulty_level = max(0.0, self.difficulty_level - 0.1)
        
        print(f"Difficulty updated to {self.difficulty_level:.2f} (Success rate: {success_rate:.2%})")
    
    def __getattr__(self, name):
        """代理到基础环境的属性"""
        return getattr(self.base_env, name)