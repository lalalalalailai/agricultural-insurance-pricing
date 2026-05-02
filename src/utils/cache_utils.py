import os
import json
import pickle
from datetime import datetime
from typing import Any, Optional

# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 缓存过期时间（秒）
CACHE_EXPIRY = 172800  # 2天

def get_cache_key(prefix: str, symbol: str, **kwargs) -> str:
    """生成缓存键"""
    key_parts = [prefix, symbol]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    return "_".join(key_parts)

def get_cache_path(key: str) -> str:
    """获取缓存文件路径"""
    return os.path.join(CACHE_DIR, f"{key}.pkl")

def save_cache(key: str, data: Any) -> None:
    """保存数据到缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path(key)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'data': data,
            'timestamp': datetime.now().timestamp()
        }, f)

def load_cache(key: str) -> Optional[Any]:
    """从缓存加载数据"""
    cache_path = get_cache_path(key)
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 检查缓存是否过期
        timestamp = cache_data.get('timestamp', 0)
        if datetime.now().timestamp() - timestamp > CACHE_EXPIRY:
            os.remove(cache_path)
            return None
        
        return cache_data.get('data')
    except Exception:
        # 缓存文件损坏，删除它
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return None

def clear_cache() -> None:
    """清除所有缓存"""
    for file in os.listdir(CACHE_DIR):
        if file.endswith('.pkl'):
            os.remove(os.path.join(CACHE_DIR, file))

def clear_expired_cache() -> None:
    """清除过期缓存"""
    for file in os.listdir(CACHE_DIR):
        if file.endswith('.pkl'):
            cache_path = os.path.join(CACHE_DIR, file)
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                timestamp = cache_data.get('timestamp', 0)
                if datetime.now().timestamp() - timestamp > CACHE_EXPIRY:
                    os.remove(cache_path)
            except Exception:
                os.remove(cache_path)