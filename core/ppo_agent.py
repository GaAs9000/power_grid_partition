"""
PPO智能体模块
包含Actor网络、Critic网络和增强的PPO算法实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from collections import defaultdict, deque
import copy


class HierarchicalActor(nn.Module):
    """
    分层策略网络，降低动作空间复杂度
    """
    
    def __init__(self, 
                 node_embedding_dim: int,
                 hidden_dim: int = 128,
                 num_regions: int = 3,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        初始化分层Actor网络
        
        Args:
            node_embedding_dim: 节点嵌入维度
            hidden_dim: 隐藏层维度
            num_regions: 区域数量
            num_layers: 网络层数
            dropout: dropout率
        """
        super(HierarchicalActor, self).__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.num_regions = num_regions
        
        # 第一层：节点选择器
        node_selector_input = node_embedding_dim * 2 + node_embedding_dim + 1 + num_regions
        
        self.node_selector = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.node_selector.add_module(f'linear_{i}', 
                    nn.Linear(node_selector_input, hidden_dim))
            else:
                self.node_selector.add_module(f'linear_{i}', 
                    nn.Linear(hidden_dim, hidden_dim))
            self.node_selector.add_module(f'relu_{i}', nn.ReLU())
            self.node_selector.add_module(f'dropout_{i}', nn.Dropout(dropout))
        self.node_selector.add_module('output', nn.Linear(hidden_dim, 1))
        
        # 第二层：区域选择器
        region_selector_input = (node_embedding_dim * 2 + 
                                num_regions * node_embedding_dim + 
                                node_embedding_dim + 1 + num_regions)
        
        self.region_selector = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.region_selector.add_module(f'linear_{i}', 
                    nn.Linear(region_selector_input, hidden_dim))
            else:
                self.region_selector.add_module(f'linear_{i}', 
                    nn.Linear(hidden_dim, hidden_dim))
            self.region_selector.add_module(f'relu_{i}', nn.ReLU())
            self.region_selector.add_module(f'dropout_{i}', nn.Dropout(dropout))
        self.region_selector.add_module('output', nn.Linear(hidden_dim, num_regions))
    
    def forward(self, state: Dict[str, torch.Tensor], 
                valid_actions: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（分层决策）
        
        Args:
            state: 环境状态
            valid_actions: 有效动作列表
        
        Returns:
            action_probs: 动作概率分布
            action_logits: 动作的原始分数
        """
        if len(valid_actions) == 0:
            return torch.tensor([]), torch.tensor([])
        
        node_embeddings = state['node_embeddings']
        region_embeddings = state['region_embeddings']
        global_context = state['global_context']
        
        # 提取有效节点
        valid_nodes = list(set([action[0] for action in valid_actions]))
        
        # 第一步：选择节点
        node_scores = []
        for node_idx in valid_nodes:
            node_feature = torch.cat([
                node_embeddings[node_idx],
                node_embeddings[valid_nodes].mean(dim=0),  # 边界节点平均
                global_context
            ])
            score = self.node_selector(node_feature)
            node_scores.append((node_idx, score))
        
        # 使用Gumbel-Softmax采样节点
        node_logits = torch.cat([score for _, score in node_scores])
        node_probs = F.softmax(node_logits, dim=0)
        
        # 第二步：为每个节点计算区域概率
        action_scores = []
        for i, (node_idx, node_score) in enumerate(node_scores):
            # 获取该节点的有效区域
            valid_regions = [action[1] for action in valid_actions if action[0] == node_idx]
            
            # 计算区域特征
            region_feature = torch.cat([
                node_embeddings[node_idx],
                region_embeddings.flatten(),
                global_context,
                node_embeddings[node_idx]  # 重复节点特征以强调
            ])
            
            region_logits = self.region_selector(region_feature)
            
            # 只保留有效区域的分数
            for region in valid_regions:
                combined_score = node_score + region_logits[region - 1]
                action_scores.append(combined_score)
        
        action_logits = torch.cat(action_scores)
        action_probs = F.softmax(action_logits, dim=0)
        
        return action_probs, action_logits


class Actor(nn.Module):
    """
    标准策略网络
    """
    
    def __init__(self, 
                 node_embedding_dim: int,
                 hidden_dim: int = 128,
                 num_regions: int = 3,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        初始化Actor网络
        """
        super(Actor, self).__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.num_regions = num_regions
        
        # 输入维度
        input_dim = (node_embedding_dim * 2 + 
                     num_regions * node_embedding_dim + 
                     node_embedding_dim + 1 + num_regions * 2 + 2)
        
        # 构建网络
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.action_scorer = nn.Sequential(*layers)
    
    def forward(self, state: Dict[str, torch.Tensor], 
                valid_actions: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        if len(valid_actions) == 0:
            return torch.tensor([]), torch.tensor([])
        
        node_embeddings = state['node_embeddings']
        region_embeddings = state['region_embeddings']
        global_context = state['global_context']
        
        action_scores = []
        
        for node_idx, region in valid_actions:
            # 构造动作特征
            node_emb = node_embeddings[node_idx]
            region_emb = region_embeddings[region - 1]
            
            # 拼接特征
            action_features = torch.cat([
                node_emb,
                region_emb,
                region_embeddings.flatten(),
                global_context
            ])
            
            # 计算分数
            score = self.action_scorer(action_features)
            action_scores.append(score)
        
        action_logits = torch.cat(action_scores)
        action_probs = F.softmax(action_logits, dim=0)
        
        return action_probs, action_logits


class Critic(nn.Module):
    """
    价值网络，估计状态价值
    """
    
    def __init__(self, 
                 node_embedding_dim: int,
                 num_regions: int = 3,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """初始化Critic网络"""
        super(Critic, self).__init__()
        
        # 输入维度：增加更多统计信息
        input_dim = (num_regions * node_embedding_dim + 
                     node_embedding_dim + 1 + num_regions * 2 + 2 +
                     num_regions * 3)  # 每个区域的额外统计
        
        # 构建网络
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.value_net = nn.Sequential(*layers)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        region_embeddings = state['region_embeddings']
        global_context = state['global_context']
        z = state['z']
        
        # 计算详细的区域统计信息
        region_stats = []
        for k in range(1, region_embeddings.shape[0] + 1):
            mask = (z == k)
            count = mask.sum().float()
            ratio = count / z.shape[0]
            
            # 添加更多统计信息
            if hasattr(state, 'Pd_pu'):
                load = state['Pd_pu'][mask].sum() if mask.any() else 0
                region_stats.extend([count, ratio, load])
            else:
                region_stats.extend([count, ratio, 0])
        
        region_stats = torch.tensor(region_stats, device=region_embeddings.device)
        
        # 拼接状态特征
        state_features = torch.cat([
            region_embeddings.flatten(),
            global_context,
            region_stats
        ])
        
        value = self.value_net(state_features)
        
        return value


class PrioritizedPPOMemory:
    """
    优先经验回放内存
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        初始化优先经验回放
        
        Args:
            capacity: 容量
            alpha: 优先级指数
            beta: 重要性采样指数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.00001
        
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.action_probs = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.valid_actions_list = deque(maxlen=capacity)
        
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, state, action, action_prob, reward, done, valid_actions):
        """添加经验"""
        self.states.append(copy.deepcopy(state))
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.valid_actions_list.append(valid_actions.copy())
        
        # 新经验给予最高优先级
        self.priorities.append(self.max_priority)
    
    def sample_batch(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """基于优先级采样批次"""
        if len(self.states) == 0:
            return [], np.array([]), np.array([])
        
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.states), batch_size, p=probs)
        
        # 计算重要性采样权重
        total = len(self.states)
        self.beta = np.min([1., self.beta + self.beta_increment])
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 获取批次数据
        batch = []
        for idx in indices:
            batch.append({
                'state': self.states[idx],
                'action': self.actions[idx],
                'action_prob': self.action_probs[idx],
                'reward': self.rewards[idx],
                'done': self.dones[idx],
                'valid_actions': self.valid_actions_list[idx]
            })
        
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级"""
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def clear(self):
        """清空经验池"""
        self.states.clear()
        self.actions.clear()
        self.action_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.valid_actions_list.clear()
        self.priorities.clear()
        self.max_priority = 1.0
    
    def __len__(self):
        return len(self.states)


class CuriosityModule(nn.Module):
    """
    好奇心驱动的探索模块
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """初始化好奇心模块"""
        super(CuriosityModule, self).__init__()
        
        # 前向模型：预测下一状态
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 逆向模型：从状态对预测动作
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def compute_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor, 
                                next_state: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """计算内在奖励"""
        # 前向预测
        state_action = torch.cat([state, action])
        pred_next_state = self.forward_model(state_action)
        forward_error = F.mse_loss(pred_next_state, next_state)
        
        # 逆向预测
        state_pair = torch.cat([state, next_state])
        pred_action = self.inverse_model(state_pair)
        inverse_error = F.mse_loss(pred_action, action)
        
        # 内在奖励 = 预测误差（鼓励探索新状态）
        intrinsic_reward = forward_error.detach().item()
        
        # 总损失用于训练
        total_loss = forward_error + inverse_error
        
        return intrinsic_reward, total_loss


class PPOAgent:
    """
    增强的PPO智能体
    """
    
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 use_curiosity: bool = False,
                 use_prioritized_replay: bool = False,
                 device: str = 'cpu'):
        """初始化PPO智能体"""
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        
        self.optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 使用优先经验回放或标准经验池
        if use_prioritized_replay:
            self.memory = PrioritizedPPOMemory()
        else:
            self.memory = PPOMemory()
        
        # 好奇心模块
        self.use_curiosity = use_curiosity
        if use_curiosity:
            # TODO: 需要定义state_dim和action_dim
            self.curiosity = CuriosityModule(state_dim=128, action_dim=10).to(self.device)
            self.optimizer_curiosity = optim.Adam(self.curiosity.parameters(), lr=1e-4)
        
        # 训练统计
        self.training_stats = defaultdict(list)
    
    def select_action(self, state: Dict[str, torch.Tensor], 
                     valid_actions: List[Tuple[int, int]], 
                     training: bool = True) -> Optional[Tuple[int, int]]:
        """选择动作"""
        if len(valid_actions) == 0:
            return None
        
        with torch.no_grad():
            action_probs, _ = self.actor(state, valid_actions)
            
            if training:
                # 动态调整探索
                entropy = -(action_probs * action_probs.log()).sum()
                
                # 如果熵太低，增加探索
                if entropy < 0.1:
                    temperature = 1.5
                    action_probs = F.softmax(action_probs.log() / temperature, dim=0)
                
                dist = Categorical(action_probs)
                action_idx = dist.sample()
            else:
                action_idx = action_probs.argmax()
            
            selected_action = valid_actions[action_idx]
            
            # 存储到经验池
            if training:
                if hasattr(self.memory, 'add'):
                    self.memory.add(
                        state=state,
                        action=action_idx,
                        action_prob=action_probs[action_idx],
                        reward=0,
                        done=False,
                        valid_actions=valid_actions
                    )
            
            return selected_action
    
    def update_memory(self, reward: float, done: bool, 
                     intrinsic_reward: float = 0.0):
        """更新最后一个经验的奖励"""
        if len(self.memory) > 0:
            total_reward = reward + intrinsic_reward * 0.1  # 内在奖励权重
            self.memory.rewards[-1] = total_reward
            self.memory.dones[-1] = done
    
    def compute_gae(self, values: torch.Tensor, rewards: List[float], 
                    dones: List[bool], next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计(GAE)"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=self.device)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self):
        """PPO更新（增强版）"""
        if len(self.memory) == 0:
            return
        
        # 如果使用优先经验回放
        if isinstance(self.memory, PrioritizedPPOMemory):
            self._update_with_priority()
        else:
            self._update_standard()
    
    def _update_standard(self):
        """标准PPO更新"""
        # 计算所有状态的价值
        old_values = []
        for state in self.memory.states:
            value = self.critic(state)
            old_values.append(value)
        old_values = torch.cat(old_values)
        
        # 计算最后一个状态的价值
        with torch.no_grad():
            last_value = self.critic(self.memory.states[-1])
        
        # 计算GAE和回报
        advantages, returns = self.compute_gae(
            old_values, self.memory.rewards, self.memory.dones, last_value
        )
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新多个epoch
        for epoch in range(self.k_epochs):
            epoch_stats = defaultdict(list)
            
            # 遍历所有经验
            for i in range(len(self.memory)):
                state = self.memory.states[i]
                action_idx = self.memory.actions[i]
                old_action_prob = self.memory.action_probs[i]
                valid_actions = self.memory.valid_actions_list[i]
                advantage = advantages[i]
                return_i = returns[i]
                
                # Actor更新
                action_probs, action_logits = self.actor(state, valid_actions)
                
                if len(action_probs) > 0:
                    dist = Categorical(action_probs)
                    action_prob = action_probs[action_idx]
                    entropy = dist.entropy()
                    
                    # 计算比率
                    ratio = action_prob / old_action_prob
                    
                    # 计算KL散度
                    kl_div = (old_action_prob * torch.log(old_action_prob / action_prob)).item()
                    epoch_stats['kl_divs'].append(kl_div)
                    
                    # PPO损失
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                    
                    # Critic更新
                    value = self.critic(state)
                    critic_loss = F.mse_loss(value, return_i.unsqueeze(0))
                    
                    # 总损失
                    total_loss = actor_loss + self.value_loss_coef * critic_loss
                    
                    # 反向传播
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    total_loss.backward()
                    
                    # 梯度裁剪
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    
                    self.optimizer_actor.step()
                    self.optimizer_critic.step()
                    
                    # 记录统计
                    epoch_stats['actor_losses'].append(actor_loss.item())
                    epoch_stats['critic_losses'].append(critic_loss.item())
                    epoch_stats['entropies'].append(entropy.item())
            
            # 早停检查
            if epoch_stats['kl_divs'] and np.mean(epoch_stats['kl_divs']) > 0.02:
                break
            
            # 记录epoch统计
            for key, values in epoch_stats.items():
                if values:
                    self.training_stats[key].append(np.mean(values))
        
        # 清空经验池
        self.memory.clear()
    
    def _update_with_priority(self):
        """使用优先经验回放的PPO更新"""
        # TODO: 实现优先经验回放版本的PPO更新
        pass
    
    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'training_stats': dict(self.training_stats)
        }
        
        if self.use_curiosity:
            checkpoint['curiosity_state_dict'] = self.curiosity.state_dict()
            checkpoint['optimizer_curiosity_state_dict'] = self.optimizer_curiosity.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.training_stats = defaultdict(list, checkpoint.get('training_stats', {}))
        
        if self.use_curiosity and 'curiosity_state_dict' in checkpoint:
            self.curiosity.load_state_dict(checkpoint['curiosity_state_dict'])
            self.optimizer_curiosity.load_state_dict(checkpoint['optimizer_curiosity_state_dict'])


class ConstrainedPPOAgent(PPOAgent):
    """
    带约束的PPO智能体
    """
    
    def __init__(self, *args, num_constraints: int = 3, **kwargs):
        """初始化约束PPO"""
        super().__init__(*args, **kwargs)
        
        self.num_constraints = num_constraints
        
        # 约束价值函数
        constraint_input_dim = kwargs.get('node_embedding_dim', 64) * 4
        self.constraint_critic = nn.Sequential(
            nn.Linear(constraint_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_constraints)
        ).to(self.device)
        
        self.optimizer_constraint = optim.Adam(
            self.constraint_critic.parameters(), 
            lr=kwargs.get('lr_critic', 1e-3)
        )
        
        # 拉格朗日乘子
        self.lagrange_multipliers = torch.zeros(num_constraints, device=self.device)
        self.constraint_thresholds = torch.tensor([0.1, 0.2, 0.15], device=self.device)
        self.alpha = 0.01  # 拉格朗日更新率
    
    def compute_constraint_violations(self, state: Dict) -> torch.Tensor:
        """计算约束违反程度"""
        # 简化的约束特征
        constraint_features = torch.cat([
            state['region_embeddings'].flatten(),
            state['global_context']
        ])
        
        violations = self.constraint_critic(constraint_features)
        return violations
    
    def update(self):
        """带约束的PPO更新"""
        if len(self.memory) == 0:
            return
        
        # 先执行标准PPO更新
        super().update()
        
        # 更新约束相关参数
        constraint_violations = []
        for state in self.memory.states:
            violations = self.compute_constraint_violations(state)
            constraint_violations.append(violations)
        
        if constraint_violations:
            avg_violations = torch.stack(constraint_violations).mean(dim=0)
            
            # 更新拉格朗日乘子
            self.lagrange_multipliers += self.alpha * F.relu(
                avg_violations - self.constraint_thresholds
            )
            self.lagrange_multipliers = torch.clamp(self.lagrange_multipliers, min=0, max=10)


class AdaptivePPOAgent(PPOAgent):
    """
    自适应PPO智能体
    """
    
    def __init__(self, *args, **kwargs):
        """初始化自适应PPO"""
        super().__init__(*args, **kwargs)
        
        self.kl_target = 0.02
        self.eps_clip_min = 0.1
        self.eps_clip_max = 0.3
        
        # 性能历史
        self.performance_history = deque(maxlen=100)
        self.kl_history = deque(maxlen=100)
        
        # 自适应学习率调度器
        self.scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_actor, mode='max', factor=0.5, patience=10
        )
        self.scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_critic, mode='max', factor=0.5, patience=10
        )
    
    def update(self):
        """自适应PPO更新"""
        if len(self.memory) == 0:
            return
        
        # 记录更新前的策略
        old_actor_params = {name: param.clone() 
                           for name, param in self.actor.named_parameters()}
        
        # 执行标准更新
        super().update()
        
        # 计算KL散度
        kl_div = self._compute_policy_kl(old_actor_params)
        self.kl_history.append(kl_div)
        
        # 自适应调整clip参数
        if kl_div > 1.5 * self.kl_target:
            self.eps_clip *= 0.8
        elif kl_div < 0.5 * self.kl_target:
            self.eps_clip *= 1.2
        
        self.eps_clip = np.clip(self.eps_clip, self.eps_clip_min, self.eps_clip_max)
        
        # 自适应调整熵系数
        if hasattr(self, 'training_stats') and 'entropies' in self.training_stats:
            recent_entropy = np.mean(self.training_stats['entropies'][-10:])
            if recent_entropy < 0.1:
                self.entropy_coef *= 1.1
            elif recent_entropy > 1.0:
                self.entropy_coef *= 0.9
            
            self.entropy_coef = np.clip(self.entropy_coef, 0.001, 0.1)
    
    def _compute_policy_kl(self, old_params: Dict) -> float:
        """计算策略KL散度"""
        kl = 0.0
        for name, param in self.actor.named_parameters():
            if name in old_params:
                kl += (old_params[name] - param).pow(2).sum()
        return kl.item()
    
    def update_performance(self, episode_reward: float):
        """更新性能历史"""
        self.performance_history.append(episode_reward)
        
        # 更新学习率
        if len(self.performance_history) >= 10:
            avg_performance = np.mean(list(self.performance_history)[-10:])
            self.scheduler_actor.step(avg_performance)
            self.scheduler_critic.step(avg_performance)


class PPOMemory:
    """标准PPO经验存储"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.valid_actions_list = []
    
    def add(self, state, action, action_prob, reward, done, valid_actions):
        self.states.append(copy.deepcopy(state))
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.valid_actions_list.append(valid_actions.copy())
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.action_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.valid_actions_list.clear()
    
    def __len__(self):
        return len(self.states)