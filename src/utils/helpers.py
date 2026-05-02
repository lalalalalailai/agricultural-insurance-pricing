import time
import functools
import logging
import os
from typing import Callable, Any

def timer(func: Callable = None, *, verbose: bool = True):
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = fn(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            if verbose:
                print(f"[TIMER] {fn.__name__} executed in {elapsed:.2f}s")
            return result
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator

def logger_setup(name: str = 'agri_pricing', level: int = logging.INFO,
                 log_file: str = None) -> logging.Logger:
    try:
        from utils.debug_manager import setup_structured_logger
        return setup_structured_logger(name, level=level, to_file=bool(log_file))
    except ImportError:
        pass
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def format_number(x: float, decimals: int = 2, suffix: bool = False) -> str:
    if abs(x) >= 1e8:
        result = f'{x/1e8:.{decimals}f}亿'
    elif abs(x) >= 1e4:
        result = f'{x/1e4:.{decimals}f}万'
    else:
        result = f'{x:.{decimals}f}'
    return result

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
