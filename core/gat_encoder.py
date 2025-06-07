"""
GAT编码器模块
使用图注意力网络学习节点嵌入，包含物理引导和多尺度版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
import numpy as np


class GATEncoder(nn.Module):
    """
    基于图注意力网络 (GAT) 的编码器，用于从图数据中学习节点嵌入
    包含残差连接、层归一化和自适应特性
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 heads: int = 8,
                 dropout: float = 0.6,
                 concat_heads: bool = True,
                 add_self_loops: bool = True,
                 use_edge_attr: bool = False):
        """
        初始化GAT编码器
        
        Args:
            in_channels: 输入节点特征的维度
            hidden_channels: GAT中间层每个头的输出维度
            out_channels: 输出节点嵌入的维度
            num_layers: GAT层的数量（必须 >= 2）
            heads: 注意力头的数量
            dropout: dropout比率
            concat_heads: 是否拼接多头输出（最后一层除外）
            add_self_loops: 是否添加自环
            use_edge_attr: 是否使用边特征
        """
        super(GATEncoder, self).__init__()
        
        assert num_layers >= 2, "GAT层数必须至少为2"
        
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_edge_attr = use_edge_attr
        
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.res_projections = nn.ModuleList()
        
        # 学习型残差权重
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_layers)
        ])
        
        # 自适应dropout
        self.dropout_schedulers = nn.ModuleList([
            nn.Linear(hidden_channels if i > 0 else in_channels, 1) 
            for i in range(num_layers)
        ])
        
        current_dim = in_channels
        
        # 边特征编码器
        if use_edge_attr:
            self.edge_encoder = nn.Sequential(
                nn.Linear(10, hidden_channels),  # 假设边特征维度为10
                nn.ReLU(),
                nn.Linear(hidden_channels, heads)
            )
        
        # 构建GAT层
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            
            if is_last_layer:
                gat_out_channels = out_channels
                concat = False
            else:
                gat_out_channels = hidden_channels
                concat = concat_heads
            
            self.gat_layers.append(
                GATConv(
                    in_channels=current_dim,
                    out_channels=gat_out_channels,
                    heads=heads,
                    dropout=self.dropout_rate,
                    concat=concat,
                    add_self_loops=add_self_loops
                )
            )
            
            # 计算实际输出维度
            if concat:
                dim_after_gat = gat_out_channels * heads
            else:
                dim_after_gat = gat_out_channels
            
            self.layer_norms.append(nn.LayerNorm(dim_after_gat))
            
            # 残差投影层
            if current_dim != dim_after_gat:
                self.res_projections.append(nn.Linear(current_dim, dim_after_gat))
            else:
                self.res_projections.append(nn.Identity())
            
            current_dim = dim_after_gat
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征矩阵 [num_nodes, in_channels]
            edge_index: 边索引矩阵 [2, num_edges]
            edge_attr: 边特征矩阵 [num_edges, edge_dim]
        
        Returns:
            节点嵌入矩阵 [num_nodes, out_channels]
        """
        # 处理边特征
        edge_weights = None
        if self.use_edge_attr and edge_attr is not None:
            edge_weights = self.edge_encoder(edge_attr)
        
        for i in range(self.num_layers):
            h_input = x
            
            # GAT卷积
            if edge_weights is not None and i == 0:
                # 只在第一层使用边权重
                x_gat = self.gat_layers[i](h_input, edge_index, edge_attr=edge_weights)
            else:
                x_gat = self.gat_layers[i](h_input, edge_index)
            
            # 层归一化
            x_norm = self.layer_norms[i](x_gat)
            
            # 学习型残差连接
            projected_input = self.res_projections[i](h_input)
            residual_weight = torch.sigmoid(self.residual_weights[i])
            x_res = x_norm + residual_weight * projected_input
            
            # 自适应dropout和激活
            if i < self.num_layers - 1:
                x = F.elu(x_res)
                
                # 计算自适应dropout率
                dropout_rate = torch.sigmoid(
                    self.dropout_schedulers[i](x.mean(dim=0, keepdim=True))
                ).item()
                x = F.dropout(x, p=dropout_rate * self.dropout_rate, training=self.training)
            else:
                x = x_res
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor, 
                            layer_idx: int = -1):
        """
        获取指定层的注意力权重（用于可视化）
        """
        with torch.no_grad():
            for i in range(self.num_layers):
                if i == layer_idx or (layer_idx == -1 and i == self.num_layers - 1):
                    _, (edge_index_with_attention, attention_weights) = self.gat_layers[i](
                        x, edge_index, return_attention_weights=True
                    )
                    return edge_index_with_attention, attention_weights
                
                # 正常前向传播
                h_input = x
                x_gat = self.gat_layers[i](h_input, edge_index)
                x_norm = self.layer_norms[i](x_gat)
                projected_input = self.res_projections[i](h_input)
                x_res = x_norm + projected_input
                
                if i < self.num_layers - 1:
                    x = F.elu(x_res)
                else:
                    x = x_res
        
        return None, None


class PhysicsGuidedGATEncoder(GATEncoder):
    """
    物理引导的GAT编码器
    """
    
    def __init__(self, *args, admittance_weight: float = 0.5, **kwargs):
        """
        初始化物理引导的GAT编码器
        
        Args:
            admittance_weight: 导纳权重的影响因子
        """
        super().__init__(*args, **kwargs)
        
        self.admittance_weight = admittance_weight
        
        # 物理感知的注意力调制器
        self.physics_modulator = nn.Sequential(
            nn.Linear(kwargs.get('edge_attr_dim', 10), self.gat_layers[0].heads),
            nn.Sigmoid()
        )
        
        # 拓扑感知的位置编码
        self.use_laplacian_pe = True
        if self.use_laplacian_pe:
            pe_dim = 16
            self.laplacian_pe_proj = nn.Linear(pe_dim, args[0])  # in_channels
    
    def compute_laplacian_pe(self, edge_index: torch.Tensor, num_nodes: int, k: int = 16):
        """计算拉普拉斯位置编码"""
        # 构建度矩阵
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 构建归一化拉普拉斯矩阵
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 计算特征值和特征向量（简化版）
        # 实际实现中应该使用更高效的方法
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj_matrix[row, col] = edge_weight
        
        # 拉普拉斯矩阵 L = I - A
        L = torch.eye(num_nodes, device=edge_index.device) - adj_matrix
        
        # 计算前k个特征向量
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            pe = eigenvectors[:, :k]
        except:
            # 如果特征分解失败，使用随机初始化
            pe = torch.randn(num_nodes, k, device=edge_index.device) * 0.1
        
        return pe
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        物理引导的前向传播
        """
        # 添加拉普拉斯位置编码
        if self.use_laplacian_pe:
            pe = self.compute_laplacian_pe(edge_index, x.shape[0])
            pe_features = self.laplacian_pe_proj(pe)
            x = x + pe_features
        
        # 计算物理引导的注意力权重
        if edge_attr is not None:
            physics_weights = self.physics_modulator(edge_attr)
            
            # 将物理权重整合到GAT计算中
            for i in range(self.num_layers):
                h_input = x
                
                # 在第一层使用物理权重
                if i == 0:
                    # 修改GAT的注意力机制
                    x_gat = self._physics_guided_gat_conv(
                        self.gat_layers[i], h_input, edge_index, 
                        edge_attr, physics_weights
                    )
                else:
                    x_gat = self.gat_layers[i](h_input, edge_index)
                
                # 后续处理与基类相同
                x_norm = self.layer_norms[i](x_gat)
                projected_input = self.res_projections[i](h_input)
                residual_weight = torch.sigmoid(self.residual_weights[i])
                x_res = x_norm + residual_weight * projected_input
                
                if i < self.num_layers - 1:
                    x = F.elu(x_res)
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)
                else:
                    x = x_res
        else:
            # 没有边特征时，使用标准前向传播
            x = super().forward(x, edge_index, edge_attr)
        
        return x
    
    def _physics_guided_gat_conv(self, gat_layer, x, edge_index, edge_attr, physics_weights):
        """
        物理引导的GAT卷积
        注：这是一个简化实现，实际中需要修改GATConv内部
        """
        # 获取标准GAT输出
        out = gat_layer(x, edge_index)
        
        # 应用物理权重调制（简化版）
        # 实际实现中应该在attention计算时就整合physics_weights
        
        return out


class MultiScaleGATEncoder(nn.Module):
    """
    多尺度GAT编码器，捕捉不同尺度的图结构
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 heads: int = 8,
                 scales: List[int] = [1, 2, 3]):
        """
        初始化多尺度GAT编码器
        
        Args:
            scales: 不同的感受野尺度（跳数）
        """
        super(MultiScaleGATEncoder, self).__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # 为每个尺度创建GAT编码器
        self.scale_encoders = nn.ModuleList()
        for scale in scales:
            encoder = GATEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,  # 注意：中间输出
                num_layers=scale,  # 使用不同深度表示不同尺度
                heads=heads
            )
            self.scale_encoders.append(encoder)
        
        # 全局池化层
        self.global_pool = GlobalAttentionPooling(hidden_channels)
        
        # 融合不同尺度的特征
        fusion_input_dim = hidden_channels * self.num_scales + hidden_channels  # +全局特征
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, out_channels)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        """
        多尺度前向传播
        """
        scale_features = []
        
        # 计算不同尺度的特征
        for i, (scale, encoder) in enumerate(zip(self.scales, self.scale_encoders)):
            # 对于更大的尺度，可以使用k跳邻居
            if scale > 1:
                # 简化：这里仍使用原始edge_index
                # 实际实现中应该计算k跳邻居的edge_index
                scale_edge_index = self._compute_k_hop_edges(edge_index, x.shape[0], k=scale)
            else:
                scale_edge_index = edge_index
            
            # 通过对应的编码器
            scale_feat = encoder(x, scale_edge_index, edge_attr)
            scale_features.append(scale_feat)
        
        # 全局特征
        global_feat = self.global_pool(scale_features[0], batch)
        
        # 扩展全局特征到每个节点
        if batch is None:
            global_feat_expanded = global_feat.repeat(x.shape[0], 1)
        else:
            global_feat_expanded = global_feat[batch]
        
        # 融合所有尺度的特征
        multi_scale_features = torch.cat(scale_features + [global_feat_expanded], dim=-1)
        
        # 最终融合
        out = self.fusion_layer(multi_scale_features)
        
        return out
    
    def _compute_k_hop_edges(self, edge_index: torch.Tensor, num_nodes: int, k: int):
        """
        计算k跳邻居的边（简化版）
        实际实现中应该使用更高效的算法
        """
        if k == 1:
            return edge_index
        
        # 构建邻接矩阵
        adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # 计算k跳邻接矩阵
        k_hop_adj = adj.clone()
        for _ in range(k - 1):
            k_hop_adj = torch.matmul(k_hop_adj, adj)
        
        # 二值化并移除自环
        k_hop_adj = (k_hop_adj > 0).float()
        k_hop_adj.fill_diagonal_(0)
        
        # 转换回edge_index格式
        k_hop_edges = k_hop_adj.nonzero().t()
        
        return k_hop_edges


class GlobalAttentionPooling(nn.Module):
    """
    全局注意力池化层
    """
    
    def __init__(self, hidden_dim: int):
        super(GlobalAttentionPooling, self).__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        计算全局池化特征
        
        Args:
            x: 节点特征 [num_nodes, hidden_dim]
            batch: 批次索引 [num_nodes]
        
        Returns:
            全局特征 [batch_size, hidden_dim] 或 [1, hidden_dim]
        """
        # 计算注意力权重
        attention_scores = self.attention_net(x)
        
        if batch is None:
            # 单图情况
            attention_weights = F.softmax(attention_scores, dim=0)
            global_feat = (x * attention_weights).sum(dim=0, keepdim=True)
        else:
            # 批处理情况
            attention_weights = F.softmax(attention_scores, dim=0)
            global_feat = global_mean_pool(x * attention_weights, batch)
        
        return global_feat


class AdaptiveGATEncoder(GATEncoder):
    """
    自适应GAT编码器，可以动态调整架构
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 动态架构参数
        self.min_heads = 4
        self.max_heads = 12
        self.current_heads = kwargs.get('heads', 8)
        
        # 性能监控
        self.performance_history = deque(maxlen=50)
        self.adaptation_interval = 10
        self.adaptation_counter = 0
    
    def adapt_architecture(self, performance_metric: float):
        """
        根据性能动态调整架构
        
        Args:
            performance_metric: 性能指标（如验证损失）
        """
        self.performance_history.append(performance_metric)
        self.adaptation_counter += 1
        
        if self.adaptation_counter % self.adaptation_interval == 0 and len(self.performance_history) > 20:
            # 计算性能趋势
            recent_perf = list(self.performance_history)[-10:]
            older_perf = list(self.performance_history)[-20:-10]
            
            avg_recent = np.mean(recent_perf)
            avg_older = np.mean(older_perf)
            
            # 如果性能下降，增加模型复杂度
            if avg_recent > avg_older * 1.05:  # 5%的恶化阈值
                self._increase_complexity()
            # 如果性能稳定，可以尝试减少复杂度
            elif avg_recent < avg_older * 0.98:  # 2%的改善
                self._decrease_complexity()
    
    def _increase_complexity(self):
        """增加模型复杂度"""
        if self.current_heads < self.max_heads:
            self.current_heads += 2
            print(f"Increasing attention heads to {self.current_heads}")
            # 注意：实际实现中需要重新初始化GAT层
    
    def _decrease_complexity(self):
        """减少模型复杂度"""
        if self.current_heads > self.min_heads:
            self.current_heads -= 1
            print(f"Decreasing attention heads to {self.current_heads}")


class LaplacianPositionalEncoding(nn.Module):
    """
    拉普拉斯位置编码模块
    """
    
    def __init__(self, pos_enc_dim: int, max_freqs: int = 10):
        super(LaplacianPositionalEncoding, self).__init__()
        self.pos_enc_dim = pos_enc_dim
        self.max_freqs = max_freqs
    
    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        计算拉普拉斯位置编码
        
        Args:
            edge_index: 边索引
            num_nodes: 节点数
        
        Returns:
            位置编码 [num_nodes, pos_enc_dim]
        """
        # 构建归一化拉普拉斯矩阵
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 边权重
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 稀疏拉普拉斯矩阵
        L = torch.sparse_coo_tensor(
            edge_index, 
            edge_weight, 
            (num_nodes, num_nodes),
            device=edge_index.device
        )
        
        # 转换为密集矩阵（注意：对于大图可能需要其他方法）
        L_dense = L.to_dense()
        L_dense = torch.eye(num_nodes, device=edge_index.device) - L_dense
        
        # 特征分解
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)
            
            # 选择前k个最小的非零特征值对应的特征向量
            idx = eigenvalues.argsort()[1:self.max_freqs+1]  # 跳过0特征值
            selected_eigenvectors = eigenvectors[:, idx]
            
            # 如果特征向量不够，用零填充
            if selected_eigenvectors.shape[1] < self.pos_enc_dim:
                padding = torch.zeros(
                    num_nodes, 
                    self.pos_enc_dim - selected_eigenvectors.shape[1],
                    device=edge_index.device
                )
                pe = torch.cat([selected_eigenvectors, padding], dim=1)
            else:
                pe = selected_eigenvectors[:, :self.pos_enc_dim]
        except:
            # 如果特征分解失败，返回随机初始化
            pe = torch.randn(num_nodes, self.pos_enc_dim, device=edge_index.device) * 0.01
        
        return pe