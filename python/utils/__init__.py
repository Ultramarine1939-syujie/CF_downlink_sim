"""
工具函数模块
"""

import os
import json
import hashlib
from pathlib import Path


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def get_file_hash(filepath):
    """计算文件的 MD5 哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_model_info(model, filepath, extra_info=None):
    """保存模型元信息"""
    info = {
        'model_type': model.__class__.__name__,
        'model_config': {
            'L': getattr(model, 'L', None),
            'K': getattr(model, 'K', None),
            'hidden_dim': getattr(model, 'hidden_dim', None),
        },
        'n_params': sum(p.numel() for p in model.parameters()),
    }
    if extra_info:
        info.update(extra_info)

    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)


def load_config(config_path):
    """加载 JSON 配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config, config_path):
    """保存配置为 JSON"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


class MetricTracker:
    """指标追踪器"""

    def __init__(self):
        self.history = {}
        self.current = {}

    def update(self, metrics):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            self.current[key] = value

    def get_best(self, key):
        """获取最佳值"""
        if key not in self.history:
            return None
        return min(self.history[key])

    def get_latest(self, key):
        """获取最新值"""
        if key not in self.history:
            return None
        return self.history[key][-1]

    def reset(self):
        """重置追踪器"""
        self.history = {}
        self.current = {}
