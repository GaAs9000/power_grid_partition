"""
配置参数
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class VizConfig:
    """可视化配置"""
    DEFAULT_FIGSIZE: Tuple[int, int] = (12, 8)
    DEFAULT_NODE_SIZE: int = 300
    MAX_NODE_SIZE: int = 1500
    DEFAULT_EDGE_ALPHA: float = 0.2
    INTER_REGION_EDGE_COLOR: str = 'red'
    INTER_REGION_EDGE_WIDTH: float = 2.0
    INTER_REGION_EDGE_ALPHA: float = 0.6
    
@dataclass 
class MetricsConfig:
    """评估指标配置"""
    ENABLE_POWER_FLOW: bool = True
    N1_SECURITY_CHECK: bool = True
    CRITICAL_NODE_THRESHOLD: float = 2.0  # 平均负荷的倍数

@dataclass
class CompareConfig:
    """比较配置"""
    RADAR_METRICS: list = None
    
    def __post_init__(self):
        if self.RADAR_METRICS is None:
            self.RADAR_METRICS = [
                'load_cv',
                'total_coupling',
                'modularity',
                'n1_security_score',
                'critical_nodes_balance'
            ]
