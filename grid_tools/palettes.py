"""
调色板管理
"""

from itertools import cycle
from matplotlib import cm
import numpy as np
from typing import List, Union

# 预定义配色方案
COLOR_SCHEMES = {
    'default': ['#808080', '#FF6B6B', '#4ECDC4', '#45B7D1', 
                '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8C471'],
    'pastel': ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', 
               '#BAE1FF', '#FFBAF3', '#E6BAFF', '#BABFFF'],
    'vibrant': ['#FF006E', '#FB5607', '#FFBE0B', '#8338EC', 
                '#3A86FF', '#06FFB4', '#FF4365', '#00F5FF'],
}

def get_palette(n: int, scheme: str = 'default') -> List[str]:
    """
    获取n个颜色的调色板
    
    Args:
        n: 需要的颜色数量
        scheme: 配色方案名称
        
    Returns:
        颜色列表
    """
    if scheme in COLOR_SCHEMES:
        base_colors = COLOR_SCHEMES[scheme]
    else:
        # 使用matplotlib的colormap
        cmap = cm.get_cmap(scheme if scheme else 'tab20')
        base_colors = [cmap(i) for i in np.linspace(0, 1, min(n, 20))]
    
    if n <= len(base_colors):
        return base_colors[:n]
    
    # 需要更多颜色时循环使用
    return [c for _, c in zip(range(n), cycle(base_colors))]

def get_continuous_palette(n: int, cmap_name: str = 'viridis') -> List[str]:
    """获取连续调色板"""
    cmap = cm.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]