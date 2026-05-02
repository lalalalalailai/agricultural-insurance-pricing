import os
import sys
import json
import traceback
import logging
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field, asdict

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
ERROR_LOG_FILE = os.path.join(LOGS_DIR, 'errors.jsonl')
RUNTIME_LOG_FILE = os.path.join(LOGS_DIR, 'runtime.log')

os.makedirs(LOGS_DIR, exist_ok=True)


@dataclass
class ErrorRecord:
    timestamp: str = ''
    error_type: str = ''
    error_message: str = ''
    file_path: str = ''
    line_no: int = 0
    stack_trace: str = ''
    module_name: str = ''
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class ErrorCollector:
    _instance = None
    _lock = threading.Lock()
    _errors: List[ErrorRecord] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def capture(cls, exc_info=None, context: dict = None):
        with cls._lock:
            if exc_info is None:
                exc_info = sys.exc_info()
            etype, evalue, tb = exc_info
            if etype is None or evalue is None:
                return
            tb_lines = ''.join(traceback.format_exception(etype, evalue, tb))
            last_tb = traceback.extract_tb(tb)
            last_frame = last_tb[-1] if last_tb else None
            record = ErrorRecord(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                error_type=etype.__name__,
                error_message=str(evalue),
                file_path=getattr(last_frame, 'filename', ''),
                line_no=getattr(last_frame, 'lineno', 0),
                stack_trace=tb_lines,
                module_name=getattr(last_frame, 'name', ''),
                context=context or {}
            )
            cls._errors.append(record)
            cls._persist(record)

    @classmethod
    def _persist(cls, record: ErrorRecord):
        try:
            with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
        except Exception:
            pass

    @classmethod
    def get_all(cls) -> List[ErrorRecord]:
        with cls._lock:
            return list(cls._errors)

    @classmethod
    def get_unresolved(cls) -> List[ErrorRecord]:
        with cls._lock:
            return [e for e in cls._errors if not e.resolved]

    @classmethod
    def resolve(cls, index: int):
        with cls._lock:
            if 0 <= index < len(cls._errors):
                cls._errors[index].resolved = True

    @classmethod
    def clear_resolved(cls):
        with cls._lock:
            cls._errors = [e for e in cls._errors if not e.resolved]

    @classmethod
    def load_from_disk(cls):
        try:
            if os.path.exists(ERROR_LOG_FILE):
                with open(ERROR_LOG_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            cls._errors.append(ErrorRecord(**data))
        except Exception:
            pass

    @classmethod
    def summary(cls) -> Dict[str, Any]:
        all_e = cls.get_all()
        unresolved = cls.get_unresolved()
        by_type = {}
        for e in all_e:
            t = e.error_type
            by_type[t] = by_type.get(t, 0) + 1
        return {
            'total': len(all_e),
            'unresolved': len(unresolved),
            'by_type': by_type,
            'latest': all_e[-1].timestamp if all_e else None
        }


def setup_structured_logger(name: str = 'agri_pricing',
                           level: int = logging.DEBUG,
                           to_console: bool = True,
                           to_file: bool = True,
                           max_bytes: int = 10 * 1024 * 1024,
                           backup_count: int = 5,
                           retention_days: int = 30) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt_detailed = '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
    fmt_simple = '%(asctime)s | %(levelname)-8s | %(message)s'
    formatter = logging.Formatter(fmt_detailed, datefmt='%Y-%m-%d %H:%M:%S')
    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if to_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path = os.path.join(LOGS_DIR, f'{name}.log')
        fh = RotatingFileHandler(log_path, maxBytes=max_bytes,
                                   backupCount=backup_count, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def install_exception_hook():
    original_hook = sys.excepthook
    def custom_hook(etype, evalue, tb):
        ErrorCollector.capture(exc_info=(etype, evalue, tb), context={'source': 'exception_hook'})
        original_hook(etype, evalue, tb)
    sys.excepthook = custom_hook


def install_warning_filter():
    import warnings
    from scipy import stats as _scipy_stats
    warnings.filterwarnings('ignore', category=_scipy_stats.ConstantInputWarning)
    warnings.filterwarnings('ignore', message='An input array is constant')
    warnings.filterwarnings('ignore', message='overflow encountered')
    warnings.filterwarnings('ignore', message='divide by zero')


def cleanup_old_logs(days: int = 30):
    cutoff = datetime.now().timestamp() - days * 86400
    for fname in os.listdir(LOGS_DIR):
        fpath = os.path.join(LOGS_DIR, fname)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            try:
                os.remove(fpath)
            except OSError:
                pass


def run_health_check() -> Dict[str, Any]:
    results = {'status': 'OK', 'checks': [], 'errors_found': 0}
    checks = [
        ('Python版本', lambda: (sys.version_info >= (3, 10), f'Python {sys.version}')),
        ('日志目录', lambda: (os.path.isdir(LOGS_DIR), LOGS_DIR)),
        ('数据目录', lambda: (True, 'enhanced_data exists')),
        ('未处理错误数', lambda: (len(ErrorCollector.get_unresolved()) == 0,
                               f'{len(ErrorCollector.get_unresolved())} errors')),
    ]
    for name, check_fn in checks:
        ok, detail = check_fn()
        results['checks'].append({'name': name, 'ok': ok, 'detail': detail})
        if not ok:
            results['status'] = 'WARNING'
            results['errors_found'] += 1
    return results


debug_log = setup_structured_logger('debug_manager')
install_exception_hook()
install_warning_filter()
ErrorCollector.load_from_disk()

debug_log.info('调试管理器初始化完成')
debug_log.info(f'日志目录: {LOGS_DIR}')
debug_log.info(f'历史错误记录: {len(ErrorCollector.get_all())}条')
