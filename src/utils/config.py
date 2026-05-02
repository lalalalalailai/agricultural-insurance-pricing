import os
import logging
from pathlib import Path
from typing import Optional
from utils.constants import DATA_CONFIG, MODEL_PARAMS

logger = logging.getLogger(__name__)

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._base_dir = Path(__file__).parent.parent.parent
        self.data_root = self._base_dir / DATA_CONFIG['futures_dir'].replace('all_factors/futures', '')
        self.model_params = MODEL_PARAMS
        self._detect_data_paths()

    def _detect_data_paths(self):
        candidates = [self._base_dir, self._base_dir.parent]

        for candidate in [self._base_dir, self._base_dir.parent]:
            data_path = candidate / 'data'
            if data_path.exists() and (data_path / 'futures').exists():
                self.data_root = data_path
                self._data_root_str = str(data_path)
                logger.info(f"✅ 检测到数据目录 (data模式): {self._data_root_str}")
                return

        for candidate in candidates:
            enhanced_path = candidate / 'enhanced_data'
            if enhanced_path.exists() and (enhanced_path / 'all_factors').exists():
                self.data_root = enhanced_path
                self._data_root_str = str(enhanced_path)
                logger.info(f"✅ 检测到数据目录 (enhanced_data模式): {self._data_root_str}")
                return

        for candidate in [self._base_dir.parent, self._base_dir.parent.parent]:
            data_dir = candidate / '05_数据集'
            if data_dir.exists() and (data_dir / 'all_factors').exists():
                self.data_root = data_dir
                self._data_root_str = str(data_dir)
                logger.info(f"✅ 检测到数据目录 (05_数据集模式): {self._data_root_str}")
                return

        self.data_root = self._base_dir
        self._data_root_str = str(self._base_dir)
        logger.warning(f"⚠️ 未检测到标准数据目录，使用默认路径: {self._data_root_str}")

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def data_root_str(self) -> str:
        return self._data_root_str

    @property
    def futures_dir(self) -> str:
        direct_path = os.path.join(self._data_root_str, 'futures')
        if os.path.exists(direct_path):
            return direct_path
        return os.path.join(self._data_root_str, 'all_factors', 'futures')

    @property
    def macro_dir(self) -> str:
        macro_direct = os.path.join(self._data_root_str, 'macro')
        if os.path.exists(macro_direct):
            return macro_direct
        return os.path.join(self._data_root_str, 'all_factors', 'macro')

    @property
    def weather_dir(self) -> str:
        return os.path.join(self._data_root_str, 'weather')

    @property
    def extended_dir(self) -> str:
        return os.path.join(self._data_root_str, 'extended_factors')

    @property
    def supplementary_dir(self) -> str:
        return os.path.join(self._data_root_str, 'supplementary_data')

    @property
    def remote_sensing_dir(self) -> str:
        rs_direct = os.path.join(self._data_root_str, 'remote_sensing')
        if os.path.exists(rs_direct):
            return rs_direct
        return os.path.join(self._data_root_str, 'all_factors', 'remote_sensing')

    def get_futures_path(self, symbol: str) -> str:
        symbol_map = {
            'A0': 'A0_豆一', 'AP0': 'AP0_苹果', 'C0': 'C0_玉米',
            'CF0': 'CF0_棉花', 'CJ0': 'CJ0_红枣', 'CS0': 'CS0_玉米淀粉',
            'EB0': 'EB0_苯乙烯', 'EG0': 'EG0_乙二醇', 'FG0': 'FG0_玻璃',
            'HC0': 'HC0_热卷', 'I0': 'I0_铁矿石', 'J0': 'J0_焦炭',
            'JD0': 'JD0_鸡蛋', 'JM0': 'JM0_焦煤', 'L0': 'L0_塑料',
            'LH0': 'LH0_生猪', 'M0': 'M0_豆粕', 'MA0': 'MA0_甲醇',
            'OI0': 'OI0_菜油', 'P0': 'P0_棕榈油', 'PG0': 'PG0_LPG',
            'PK0': 'PK0_花生', 'PP0': 'PP0_PP', 'RM0': 'RM0_菜粕',
            'RU0': 'RU0_橡胶', 'SA0': 'SA0_纯碱', 'SF0': 'SF0_硅铁',
            'SM0': 'SM0_锰硅', 'SP0': 'SP0_纸浆', 'SR0': 'SR0_白糖',
            'SS0': 'SS0_不锈钢', 'TA0': 'TA0_PTA', 'UR0': 'UR0_尿素',
            'V0': 'V0_PVC', 'Y0': 'Y0_豆油', 'ZC0': 'ZC0_动力煤'
        }
        name = symbol_map.get(symbol, symbol)
        return os.path.join(self.futures_dir, f'{name}.csv')

config = Config()
