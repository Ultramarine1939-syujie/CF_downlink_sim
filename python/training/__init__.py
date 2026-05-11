"""
训练模块

提供GLP-GNN训练功能
"""

from .train_centralized import train_centralized

__all__ = ['train_centralized', 'GLPGNN', 'GLPGNN_Slim', 'train_glp_gnn']
