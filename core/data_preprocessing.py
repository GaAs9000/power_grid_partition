"""
数据预处理模块
处理MATPOWER格式的电网数据，提取节点和边特征
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional, List
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, RobustScaler


def clean_and_extract_features(mpc: Dict, K: int = 3, 
                               normalize: bool = True,
                               add_graph_features: bool = True) -> Tuple[int, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    清理原始 MATPOWER 案例文件并提取适用于图神经网络 (GNN) 的特征
    
    Args:
        mpc: MATPOWER格式的电网数据字典
        K: 目标分区数
        normalize: 是否归一化特征
        add_graph_features: 是否添加图结构特征
    
    Returns:
        K: 目标分区数
        baseMVA: 系统基准功率
        df_XN: 节点特征DataFrame
        df_Eidx: 边索引DataFrame
        df_XE: 边特征DataFrame
    """
    baseMVA = mpc['baseMVA']
    
    # 1. 清理支路容量
    branch = mpc['branch'].copy()
    for c in [5, 6, 7]:  # rateA, rateB, rateC columns
        vals = branch[:, c]
        vals[vals == 0] = baseMVA
        branch[:, c] = vals

    # 2. 提取节点特征
    bus = mpc['bus']
    nb = bus.shape[0]
    
    # 2.1 节点类型 one-hot 编码
    types = bus[:, 1].astype(int) - 1
    bus_type_oh = np.eye(3)[types]
    
    # 2.2 负荷和并联元件
    Pd_pu = bus[:, 2] / baseMVA
    Qd_pu = bus[:, 3] / baseMVA
    Gs = bus[:, 4] / baseMVA  # 归一化
    Bs = bus[:, 5] / baseMVA  # 归一化
    
    # 2.3 电压等级归一化
    baseKV = bus[:, 9]
    max_kv = np.nanmax(baseKV[baseKV < np.inf])
    if max_kv == 0 or np.isnan(max_kv):
        max_kv = 1.0
    baseKV_norm = baseKV / max_kv
    
    # 2.4 电压限制
    Vmin = bus[:, 12]
    Vmax = bus[:, 11]
    V_range = Vmax - Vmin
    V_mid = (Vmax + Vmin) / 2
    
    # 2.5 发电机聚合信息
    gen = mpc['gen']
    is_gen = np.zeros(nb, int)
    sum_Pg = np.zeros(nb)
    sum_Qg = np.zeros(nb)
    sum_Pg_min = np.zeros(nb)
    sum_Qg_min = np.zeros(nb)
    gen_count = np.zeros(nb)
    
    for g in gen:
        bi = int(g[0]) - 1
        is_gen[bi] = 1
        gen_count[bi] += 1
        sum_Pg[bi] += g[8]  # Pmax
        sum_Qg[bi] += g[3]  # Qmax
        sum_Pg_min[bi] += g[9]  # Pmin
        sum_Qg_min[bi] += g[4]  # Qmin
    
    sum_Pg_pu = sum_Pg / baseMVA
    sum_Qg_pu = sum_Qg / baseMVA
    sum_Pg_min_pu = sum_Pg_min / baseMVA
    sum_Qg_min_pu = sum_Qg_min / baseMVA
    
    # 2.6 节点度数和图特征
    idx_f = branch[:, 0].astype(int) - 1
    idx_t = branch[:, 1].astype(int) - 1
    deg = np.bincount(np.hstack([idx_f, idx_t]), minlength=nb)
    
    # 额外的图特征
    if add_graph_features:
        # 构建邻接矩阵
        adj_matrix = np.zeros((nb, nb))
        for i in range(len(idx_f)):
            adj_matrix[idx_f[i], idx_t[i]] = 1
            adj_matrix[idx_t[i], idx_f[i]] = 1
        
        # 聚类系数
        clustering_coef = compute_clustering_coefficient(adj_matrix)
        
        # 中心性指标
        eigenvector_centrality = compute_eigenvector_centrality(adj_matrix)
        betweenness_centrality = compute_betweenness_centrality_approx(adj_matrix)
        
        # k-core数
        k_core = compute_k_core(adj_matrix)
    else:
        clustering_coef = np.zeros(nb)
        eigenvector_centrality = np.zeros(nb)
        betweenness_centrality = np.zeros(nb)
        k_core = np.zeros(nb)
    
    # 2.7 区域标识（如果有）
    area = bus[:, 6] if bus.shape[1] > 6 else np.ones(nb)
    zone = bus[:, 10] if bus.shape[1] > 10 else np.ones(nb)
    
    # 组装节点特征
    X_N = np.hstack([
        bus_type_oh,
        Pd_pu[:, None], Qd_pu[:, None],
        Gs[:, None], Bs[:, None],
        baseKV_norm[:, None],
        V_range[:, None], V_mid[:, None],
        is_gen[:, None],
        sum_Pg_pu[:, None], sum_Qg_pu[:, None],
        sum_Pg_min_pu[:, None], sum_Qg_min_pu[:, None],
        gen_count[:, None],
        deg[:, None],
        clustering_coef[:, None],
        eigenvector_centrality[:, None],
        betweenness_centrality[:, None],
        k_core[:, None]
    ])
    
    # 特征名称
    feature_names = [
        'PQ', 'PV', 'Slack', 'Pd_pu', 'Qd_pu', 'Gs', 'Bs', 'baseKV_norm',
        'V_range', 'V_mid', 'is_gen', 'sum_Pg_pu', 'sum_Qg_pu',
        'sum_Pg_min_pu', 'sum_Qg_min_pu', 'gen_count', 'degree',
        'clustering_coef', 'eigenvector_centrality', 'betweenness_centrality', 'k_core'
    ]
    
    # 归一化
    if normalize:
        scaler = RobustScaler()
        # 只归一化非one-hot编码的特征
        X_N[:, 3:] = scaler.fit_transform(X_N[:, 3:])
    
    df_XN = pd.DataFrame(X_N, columns=feature_names, index=np.arange(1, nb + 1))

    # 3. 边索引
    E_idx = np.vstack([idx_f, idx_t])
    df_Eidx = pd.DataFrame(
        E_idx.T + 1,  # 转回 1-based
        columns=['fbus', 'tbus']
    )
    
    # 4. 边特征（增强版）
    r = branch[:, 2]
    x = branch[:, 3]
    b_sh = branch[:, 4]
    
    # 支路容量
    rateA_pu = branch[:, 5] / baseMVA
    rateB_pu = branch[:, 6] / baseMVA
    rateC_pu = branch[:, 7] / baseMVA
    
    # 电气参数
    z_abs = np.sqrt(r**2 + x**2)
    conductance = r / (r**2 + x**2 + 1e-10)
    susceptance = -x / (r**2 + x**2 + 1e-10)
    
    # 相角差限制
    angle_min = branch[:, 12] * np.pi / 180 if branch.shape[1] > 12 else -np.pi * np.ones(len(branch))
    angle_max = branch[:, 11] * np.pi / 180 if branch.shape[1] > 11 else np.pi * np.ones(len(branch))
    
    # 变压器信息
    ratio = branch[:, 8]
    angle = branch[:, 9] * np.pi / 180
    is_transformer = ((ratio != 0) & (ratio != 1)).astype(int)
    transformer_ratio = ratio.copy()
    transformer_ratio[transformer_ratio == 0] = 1.0
    
    # 线路状态
    status = branch[:, 10].astype(int)
    
    # 线路长度估计（基于阻抗）
    estimated_length = z_abs * 100  # 假设单位长度阻抗
    
    # 热稳定限制指标
    thermal_limit_ratio = np.minimum(rateA_pu, np.minimum(rateB_pu, rateC_pu))
    
    X_E = np.vstack([
        r, x, z_abs, conductance, susceptance, b_sh,
        rateA_pu, rateB_pu, rateC_pu, thermal_limit_ratio,
        angle_min, angle_max,
        is_transformer, transformer_ratio, angle,
        status, estimated_length
    ]).T
    
    edge_feature_names = [
        'r', 'x', 'z_abs', 'conductance', 'susceptance', 'b_sh',
        'rateA_pu', 'rateB_pu', 'rateC_pu', 'thermal_limit_ratio',
        'angle_min', 'angle_max',
        'is_transformer', 'transformer_ratio', 'transformer_angle',
        'status', 'estimated_length'
    ]
    
    # 边特征归一化
    if normalize:
        edge_scaler = RobustScaler()
        # 不归一化二元特征
        non_binary_mask = [i for i in range(len(edge_feature_names)) 
                          if edge_feature_names[i] not in ['is_transformer', 'status']]
        X_E[:, non_binary_mask] = edge_scaler.fit_transform(X_E[:, non_binary_mask])
    
    df_XE = pd.DataFrame(X_E, columns=edge_feature_names, index=np.arange(1, X_E.shape[0] + 1))
    
    return K, baseMVA, df_XN, df_Eidx, df_XE


def compute_clustering_coefficient(adj_matrix: np.ndarray) -> np.ndarray:
    """计算每个节点的聚类系数"""
    n = adj_matrix.shape[0]
    clustering_coef = np.zeros(n)
    
    for i in range(n):
        neighbors = np.where(adj_matrix[i] > 0)[0]
        k = len(neighbors)
        
        if k < 2:
            clustering_coef[i] = 0
            continue
        
        # 计算邻居之间的连接数
        neighbor_connections = 0
        for j in range(k):
            for l in range(j + 1, k):
                if adj_matrix[neighbors[j], neighbors[l]] > 0:
                    neighbor_connections += 1
        
        clustering_coef[i] = 2 * neighbor_connections / (k * (k - 1))
    
    return clustering_coef


def compute_eigenvector_centrality(adj_matrix: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """计算特征向量中心性"""
    n = adj_matrix.shape[0]
    centrality = np.ones(n) / n
    
    for _ in range(max_iter):
        new_centrality = adj_matrix @ centrality
        norm = np.linalg.norm(new_centrality)
        if norm > 0:
            new_centrality /= norm
        
        if np.allclose(centrality, new_centrality):
            break
        
        centrality = new_centrality
    
    return centrality


def compute_betweenness_centrality_approx(adj_matrix: np.ndarray, k: int = 10) -> np.ndarray:
    """
    近似计算介数中心性（使用k个随机最短路径）
    """
    n = adj_matrix.shape[0]
    betweenness = np.zeros(n)
    
    # 随机选择k个源节点
    sources = np.random.choice(n, min(k, n), replace=False)
    
    for source in sources:
        # 使用BFS计算最短路径
        distances, paths = bfs_shortest_paths(adj_matrix, source)
        
        # 统计每个节点在最短路径中出现的次数
        for target in range(n):
            if target != source and distances[target] < float('inf'):
                path = reconstruct_path(paths, source, target)
                for node in path[1:-1]:  # 不包括源和目标
                    betweenness[node] += 1
    
    # 归一化
    betweenness /= k
    
    return betweenness


def compute_k_core(adj_matrix: np.ndarray) -> np.ndarray:
    """计算每个节点的k-core数"""
    n = adj_matrix.shape[0]
    degrees = adj_matrix.sum(axis=1)
    k_core = np.zeros(n)
    
    remaining = set(range(n))
    current_k = 0
    
    while remaining:
        # 找到度数最小的节点
        min_degree = min(degrees[list(remaining)])
        
        if min_degree > current_k:
            current_k = min_degree
        
        # 移除所有度数等于min_degree的节点
        to_remove = []
        for node in remaining:
            if degrees[node] <= min_degree:
                k_core[node] = current_k
                to_remove.append(node)
        
        # 更新剩余节点的度数
        for node in to_remove:
            remaining.remove(node)
            for neighbor in range(n):
                if neighbor in remaining and adj_matrix[node, neighbor] > 0:
                    degrees[neighbor] -= 1
    
    return k_core


def bfs_shortest_paths(adj_matrix: np.ndarray, source: int) -> Tuple[np.ndarray, Dict]:
    """BFS计算最短路径"""
    n = adj_matrix.shape[0]
    distances = np.full(n, float('inf'))
    distances[source] = 0
    paths = {source: [source]}
    
    queue = [source]
    visited = set([source])
    
    while queue:
        current = queue.pop(0)
        
        for neighbor in range(n):
            if adj_matrix[current, neighbor] > 0 and neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current] + 1
                paths[neighbor] = paths[current] + [neighbor]
                queue.append(neighbor)
    
    return distances, paths


def reconstruct_path(paths: Dict, source: int, target: int) -> List[int]:
    """重建最短路径"""
    return paths.get(target, [])


def create_pyg_data(df_XN: pd.DataFrame, df_Eidx: pd.DataFrame, df_XE: pd.DataFrame,
                   add_virtual_edges: bool = False,
                   self_loops: bool = True) -> Data:
    """
    从DataFrame创建PyTorch Geometric Data对象
    
    Args:
        df_XN: 节点特征DataFrame
        df_Eidx: 边索引DataFrame
        df_XE: 边特征DataFrame
        add_virtual_edges: 是否添加虚拟边（用于长距离依赖）
        self_loops: 是否添加自环
    
    Returns:
        PyG Data对象
    """
    # 节点特征
    x = torch.tensor(df_XN.values, dtype=torch.float)
    
    # 边索引（创建无向图）
    fbus = df_Eidx['fbus'].to_numpy(dtype=int) - 1
    tbus = df_Eidx['tbus'].to_numpy(dtype=int) - 1
    
    # 基础边
    edge_index = torch.tensor(
        np.vstack([np.concatenate([fbus, tbus]), np.concatenate([tbus, fbus])]),
        dtype=torch.long
    )
    
    # 边特征（双向复制）
    edge_attr_raw = torch.tensor(df_XE.values, dtype=torch.float)
    edge_attr = torch.cat([edge_attr_raw, edge_attr_raw], dim=0)
    
    # 添加虚拟边（可选）
    if add_virtual_edges:
        virtual_edges, virtual_attrs = create_virtual_edges(x, edge_index, k=5)
        edge_index = torch.cat([edge_index, virtual_edges], dim=1)
        edge_attr = torch.cat([edge_attr, virtual_attrs], dim=0)
    
    # 添加自环（可选）
    if self_loops:
        num_nodes = x.shape[0]
        self_loop_index = torch.stack([
            torch.arange(num_nodes),
            torch.arange(num_nodes)
        ])
        self_loop_attr = torch.zeros(num_nodes, edge_attr.shape[1])
        
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
    
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 添加额外信息
    data.num_nodes = x.shape[0]
    data.node_names = df_XN.index.tolist()
    
    return data


def create_virtual_edges(x: torch.Tensor, edge_index: torch.Tensor, 
                        k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建虚拟边以捕捉长距离依赖
    
    Args:
        x: 节点特征
        edge_index: 原始边索引
        k: 每个节点添加的虚拟边数
    
    Returns:
        virtual_edges: 虚拟边索引
        virtual_attrs: 虚拟边特征
    """
    num_nodes = x.shape[0]
    
    # 计算节点间的特征距离
    distances = torch.cdist(x, x, p=2)
    
    # 移除已存在的边
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    distances[adj_matrix > 0] = float('inf')
    distances.fill_diagonal_(float('inf'))
    
    # 为每个节点选择k个最近的非邻居节点
    virtual_edges = []
    
    for i in range(num_nodes):
        # 找到k个最近的节点
        _, nearest = torch.topk(distances[i], k, largest=False)
        
        for j in nearest:
            if distances[i, j] < float('inf'):
                virtual_edges.append([i, j.item()])
                virtual_edges.append([j.item(), i])  # 无向边
    
    if virtual_edges:
        virtual_edges = torch.tensor(virtual_edges).t()
        
        # 创建虚拟边特征（简化版：使用距离作为特征）
        virtual_attrs = torch.zeros(virtual_edges.shape[1], 17)  # 假设17个边特征
        
        # 设置一些基本特征
        for i in range(virtual_edges.shape[1]):
            src, dst = virtual_edges[:, i]
            dist = distances[src, dst]
            virtual_attrs[i, 2] = dist  # z_abs位置
            virtual_attrs[i, 15] = 0  # status = 0 表示虚拟边
    else:
        virtual_edges = torch.empty((2, 0), dtype=torch.long)
        virtual_attrs = torch.empty((0, 17))
    
    return virtual_edges, virtual_attrs


def get_node_positions(mpc: Dict, use_geographic: bool = True) -> Optional[Dict[int, Tuple[float, float]]]:
    """
    获取节点的地理位置或生成布局位置
    
    Args:
        mpc: MATPOWER格式的电网数据
        use_geographic: 是否使用地理坐标（如果可用）
    
    Returns:
        节点位置字典 {node_id: (x, y)}
    """
    bus = mpc['bus']
    n_bus = bus.shape[0]
    
    # 检查是否有地理坐标
    if use_geographic and bus.shape[1] >= 14:
        # 假设13和14列是经纬度
        lon = bus[:, 12]
        lat = bus[:, 13]
        
        # 检查坐标有效性
        if not np.all(lon == 0) and not np.all(lat == 0):
            positions = {}
            for i in range(n_bus):
                positions[i] = (lon[i], lat[i])
            return positions
    
    # 如果没有地理坐标，使用力导向布局
    return None  # 让可视化函数自己生成布局


def validate_power_grid_data(mpc: Dict) -> Dict[str, List[str]]:
    """
    验证电网数据的完整性和一致性
    
    Args:
        mpc: MATPOWER格式数据
    
    Returns:
        包含警告和错误的字典
    """
    issues = {'errors': [], 'warnings': []}
    
    # 检查必要字段
    required_fields = ['bus', 'gen', 'branch', 'baseMVA']
    for field in required_fields:
        if field not in mpc:
            issues['errors'].append(f"Missing required field: {field}")
    
    if issues['errors']:
        return issues
    
    bus = mpc['bus']
    gen = mpc['gen']
    branch = mpc['branch']
    
    # 检查节点编号
    bus_ids = set(bus[:, 0].astype(int))
    
    # 检查发电机节点
    for i, g in enumerate(gen):
        if int(g[0]) not in bus_ids:
            issues['errors'].append(f"Generator {i} connected to non-existent bus {int(g[0])}")
    
    # 检查支路连接
    for i, br in enumerate(branch):
        if int(br[0]) not in bus_ids:
            issues['errors'].append(f"Branch {i} from bus {int(br[0])} does not exist")
        if int(br[1]) not in bus_ids:
            issues['errors'].append(f"Branch {i} to bus {int(br[1])} does not exist")
    
    # 检查孤立节点
    connected_buses = set()
    for br in branch:
        connected_buses.add(int(br[0]))
        connected_buses.add(int(br[1]))
    
    isolated_buses = bus_ids - connected_buses
    if isolated_buses:
        issues['warnings'].append(f"Isolated buses found: {isolated_buses}")
    
    # 检查数据范围
    if np.any(bus[:, 2] < 0):  # Pd
        issues['warnings'].append("Negative active power loads found")
    
    if np.any(branch[:, 2] < 0):  # r
        issues['errors'].append("Negative resistance found in branches")
    
    # 检查电压限制
    if np.any(bus[:, 11] < bus[:, 12]):  # Vmax < Vmin
        issues['errors'].append("Invalid voltage limits (Vmax < Vmin)")
    
    return issues
