# Spec: 修复 "未找到品种 A0 的数据" 错误

## 📌 问题概述

**错误信息**: `❌ 未找到品种 A0 的数据，请检查数据目录`

**触发位置**: [app.py:692](src/app.py#L692)

**影响范围**: 系统无法加载任何期货品种数据，导致整个数据探索、建模、定价功能完全不可用

---

## 🔍 根因分析 (QL Brain V211 深度诊断)

### 1. 目录结构对比

```
❌ 配置期望路径:
03_源码及说明/
└── enhanced_data/
    └── all_factors/
        └── futures/          ← Config.futures_dir 指向这里
            └── A0_豆一.csv   ← 不存在!

✅ 实际数据路径:
03_源码及说明/
└── data/
    └── futures/             ← 实际数据在这里
        ├── A0_豆一.csv      ✓ 存在 (35个品种)
        ├── AP0_苹果.csv
        └── ... (共35个CSV)
```

### 2. 配置链路追踪

| 组件 | 文件 | 行为 | 问题 |
|------|------|------|------|
| **DATA_CONFIG** | constants.py:102 | `'futures_dir': 'enhanced_data/all_factors/futures'` | ❌ 硬编码错误路径 |
| **Config._detect_data_paths()** | config.py:24-39 | 查找 `enhanced_data` 或 `05_数据集` | ❌ 未检测到 `data/` 目录 |
| **Config.futures_dir** | config.py:50-51 | 返回 `{data_root}/all_factors/futures` | ❌ 路径拼接错误 |
| **Config.get_futures_path()** | config.py:73-89 | 返回 `{futures_dir}/{symbol}.csv` | ❌ 基础路径已错 |
| **DataLoader.load_futures_data()** | data_loader.py:36-71 | `os.path.exists(path)` → False | ⚠️ 正常返回空DataFrame |
| **app.py** | app.py:690-693 | `if df.empty:` → 显示错误 | ✅ 错误处理正确 |

### 3. 核心矛盾点

**问题本质**: `config.py` 的 `_detect_data_paths()` 方法只查找两种目录模式：
- 模式A: `{base}/enhanced_data/all_factors/`
- 模式B: `{parent}/05_数据集/all_factors/`

但实际项目使用的是：
- **真实模式**: `{base}/data/{futures,weather,remote_sensing,...}`

---

## 🎯 解决方案

### 方案选择: **修改配置检测逻辑** (推荐 ⭐⭐⭐)

**理由**:
- 最小改动范围
- 保持向后兼容
- 自动适应多种部署环境

#### 修改文件清单

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `src/utils/config.py` | 增强 `_detect_data_paths()` 添加 `data/` 目录检测 | P0 |
| `src/utils/constants.py` | 更新 `DATA_CONFIG` 默认值 (可选) | P1 |

#### 具体修改细节

##### 修改1: `config.py` - `_detect_data_paths()` 方法

```python
def _detect_data_paths(self):
    candidates = [self._base_dir, self._base_dir.parent]

    # 🔧 新增: 优先检测 data/ 目录 (当前项目实际结构)
    for candidate in [self._base_dir, self._base_dir.parent]:
        data_path = candidate / 'data'
        if data_path.exists() and (data_path / 'futures').exists():
            self.data_root = data_path
            self._data_root_str = str(data_path)
            logger.info(f"✅ 检测到数据目录: {self._data_root_str}")
            return

    # 原有逻辑: enhanced_data
    for candidate in candidates:
        enhanced_path = candidate / 'enhanced_data'
        if enhanced_path.exists() and (enhanced_path / 'all_factors').exists():
            self.data_root = enhanced_path
            self._data_root_str = str(enhanced_path)
            return

    # 原有逻辑: 05_数据集
    for candidate in [self._base_dir.parent, self._base_dir.parent.parent]:
        data_dir = candidate / '05_数据集'
        if data_dir.exists() and (data_dir / 'all_factors').exists():
            self.data_root = data_dir
            self._data_root_str = str(data_dir)
            return

    # 回退
    self.data_root = self._base_dir
    self._data_root_str = str(self._base_dir)
```

##### 修改2: `config.py` - 属性方法更新

```python
@property
def futures_dir(self) -> str:
    # 🔧 支持 data/ 和 all_factors/ 两种结构
    if os.path.exists(os.path.join(self._data_root_str, 'futures')):
        return os.path.join(self._data_root_str, 'futures')
    return os.path.join(self._data_root_str, 'all_factors', 'futures')

@property
def weather_dir(self) -> str:
    return os.path.join(self._data_root_str, 'weather')

# 其他属性类似调整...
```

---

## ✅ 验证标准

### 功能验证
- [ ] 选择任意品种（如 A0 豆一）能正常加载数据
- [ ] 数据预览表格显示正确（日期、OHLCV字段）
- [ ] 时间范围过滤功能正常
- [ ] 天气数据能按品种自动匹配主产区
- [ ] 遥感数据（NDVI/EVI/LST/干旱指数）可正常加载

### 边界情况测试
- [ ] 切换不同品种时数据刷新正常
- [ ] 清除缓存后重新加载成功
- [ ] 不存在的品种代码返回友好提示（非崩溃）

### 性能验证
- [ ] 首次加载 < 3秒（含缓存生成）
- [ ] 缓存命中后加载 < 500ms
- [ ] 内存占用稳定（无泄漏）

---

## 📊 影响评估

| 维度 | 影响 | 说明 |
|------|------|------|
| **用户体验** | 🔴 Critical | 当前系统完全不可用 |
| **代码改动量** | 🟢 Low | 仅修改1个核心文件（~30行） |
| **回归风险** | 🟢 Low | 改动局限于路径检测逻辑 |
| **兼容性** | 🟡 Medium | 需测试多种部署场景 |

---

## 🔄 回滚方案

如果修改引入新问题：
1. 备份原文件: `config.py.bak`
2. Git回滚: `git checkout HEAD -- src/utils/config.py`
3. 或临时硬编码: `self._data_root_str = os.path.dirname(os.path.dirname(__file__)) + '/../data'`

---

## 📝 实施约束

- ✅ 必须保持与现有缓存机制兼容
- ✅ 不能破坏 Docker 部署流程
- ✅ 需添加日志输出便于调试
- ✅ 注释清晰说明路径检测优先级

---

**Spec 版本**: v1.0
**创建时间**: 2026-04-27
**状态**: 待用户确认
**预估工时**: 15分钟（含测试）
