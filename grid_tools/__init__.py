"""
Grid Tools - 电网分区可视化和评估工具包
"""

from .config import Visualizer
from .viz_interactive import InteractiveVisualizer
from .metrics import PartitionEvaluator
from .compare import PartitionComparator
from .palettes import get_palette, COLOR_SCHEMES

# 版本信息
__version__ = "1.0.0"
__all__ = [
    "Visualizer",
    "InteractiveVisualizer", 
    "PartitionEvaluator",
    "PartitionComparator",
    "GridAnalyzer"
]

class GridAnalyzer:
    """一站式电网分析接口"""
    
    def __init__(self):
        self.viz = Visualizer()
        self.iviz = InteractiveVisualizer()
        self.evaluator = PartitionEvaluator()
        self.comparator = PartitionComparator(self.evaluator)
    
    def analyze(self, env, mpc_data=None, save_dir="results"):
        """完整分析流程"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 评估
        metrics = self.evaluator.evaluate_partition(env)
        
        # 2. 可视化
        self.viz.plot_power_grid_partition(
            env, 
            save_path=f"{save_dir}/static_partition.png"
        )
        
        if mpc_data is not None:
            self.iviz.plot_partition(
                env, mpc_data,
                save_path=f"{save_dir}/interactive_partition.html"
            )
        
        return metrics