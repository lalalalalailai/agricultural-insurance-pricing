# Tasks: 修复数据路径配置问题

**项目**: 农险期货智能定价模型
**问题**: 未找到品种 A0 的数据
**优先级**: P0 (系统完全不可用)
**预估总工时**: 20分钟

---

## 📋 任务分解 (WBS)

### Phase 1: 核心修复 (15分钟)

#### ✅ Task 1.1: 修改 `config.py` 路径检测逻辑
- **文件**: `src/utils/config.py`
- **方法**: `_detect_data_paths()`
- **改动量**: +12行代码
- **具体操作**:
  - [ ] 在现有检测逻辑前插入 `data/` 目录检测
  - [ ] 添加优先级：`data/` > `enhanced_data/` > `05_数据集/`
  - [ ] 添加日志输出: `logger.info(f"✅ 检测到数据目录: {path}")`
  - [ ] 验证 Path 对象拼接正确性
- **验收标准**:
  - ```python
    # 测试用例
    config = Config()
    assert 'data' in config.data_root_str
    assert os.path.exists(config.futures_dir)
    assert os.path.exists(f"{config.futures_dir}/A0_豆一.csv")
    ```

#### ✅ Task 1.2: 更新属性方法适配双路径结构
- **文件**: `src/utils/config.py`
- **方法**: `futures_dir`, `weather_dir`, `remote_sensing_dir` 等
- **改动量**: +15行代码
- **具体操作**:
  - [ ] 修改 `futures_dir` 属性支持 `data/futures` 和 `all_factors/futures`
  - [ ] 修改 `weather_dir` 属性（通常直接在 data_root 下）
  - [ ] 修改 `remote_sensing_dir` 属性
  - [ ] 保持 `macro_dir` 和 `extended_dir` 向后兼容
- **验收标准**:
  - 所有属性返回的路径都存在
  - 路径下包含预期的 CSV 文件

---

### Phase 2: 验证测试 (5分钟)

#### ✅ Task 2.1: 单元测试 - 数据加载
- **测试场景**:
  - [ ] 测试 A0 豆一数据加载成功
  - [ ] 测试 AP0 苹果数据加载成功
  - [ ] 测试不存在的品种返回空 DataFrame
  - [ ] 测试缓存机制正常工作
- **命令**:
  ```bash
  cd 03_源码及说明
  python -c "
  from src.utils.config import config
  from src.data.data_loader import DataLoader
  loader = DataLoader()
  df = loader.load_futures_data('A0')
  print(f'✅ 加载成功: {len(df)} 行')
  assert not df.empty, '数据不应为空'
  "
  ```

#### ✅ Task 2.2: 集成测试 - Streamlit 启动
- **测试场景**:
  - [ ] 启动 Streamlit 应用无报错
  - [ ] 侧边栏显示正确的数据根目录路径
  - [ ] 选择品种后能正常显示数据表格
  - [ ] 切换品种时数据刷新正常
- **命令**:
  ```bash
  cd 03_源码及说明
  streamlit run src/app.py --server.port 8501
  ```
- **手动验证步骤**:
  1. 打开浏览器访问 http://localhost:8501
  2. 在侧边栏选择 "豆一 (A0)"
  3. 进入"数据探索"页面
  4. 确认显示期货数据表格（非错误信息）

---

### Phase 3: 边界情况与回归 (可选, +10分钟)

#### ⚪ Task 3.1: 多环境兼容性测试
- **测试环境**:
  - [ ] Windows 本地开发 (当前)
  - [ ] Docker 容器部署
  - [ ] Linux 服务器
- **验证点**:
  - 路径分隔符 (`/` vs `\`)
  - 权限问题
  - 符号链接支持

#### ⚪ Task 3.2: 缓存清理测试
- **操作**:
  1. 点击"刷新缓存"按钮
  2. 删除 `src/cache/` 目录下的所有 .pkl 文件
  3. 重新加载数据确认性能可接受
- **预期结果**:
  - 首次加载 < 3秒
  - 后续加载 < 500ms (缓存命中)

---

## 📊 任务依赖关系

```
Task 1.1 (修改路径检测)
    ↓
Task 1.2 (更新属性) → 可并行
    ↓                ↓
Task 2.1 (单元测试)   Task 2.2 (集成测试)
    ↓                ↓
    └──────→ Task 3.x (可选测试)
```

## 🔧 风险点与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| 修改引入新路径错误 | 低 | 高 | 保留原文件备份 |
| 缓存机制失效 | 中 | 中 | 提供缓存清理脚本 |
| Docker 构建失败 | 低 | 中 | 测试 Dockerfile 兼容性 |

## ✅ 完成定义 (DoD)

- [x] spec.md 已创建并经用户确认
- [ ] `config.py` 修改完成并通过单元测试
- [ ] Streamlit 应用启动正常，A0 品种数据加载成功
- [ ] 至少测试 3 个不同品种的数据加载
- [ ] 缓存机制正常工作
- [ ] 无新增 lint 错误或类型检查警告
- [ ] (可选) Docker 部署验证通过

---

**Tasks 版本**: v1.0
**最后更新**: 2026-04-27
**负责人**: QL Brain V211 Agent
