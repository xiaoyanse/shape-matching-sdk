# 更新日志

## v0.1.0（2026-11-19）

形状模板匹配引擎 SDK 首个版本：C++17 内核 + C ABI v3 单合同 + 官方 C/C++/Python/Rust 三语绑定。

### 能力

- 模板训练（单级/两级粗精编排，角度分组+缩放档位，detector 三参可调：特征数/梯度迟滞双阈）
- 匹配：单模型、同帧多模型批量（逐项状态隔离）、多目标搜索（粗候选 16/精修窗 8 预算，早停计数与最终输出同代码路径）
- 模型资产：版本化容器（VF2S3）+旧单级回退；17 位精度配置往返；篡改防护（仅 PNG 模板段+跨段一致性+总量预算）
- 几何精修：B7 轮廓点连续精修（两级 top1）
- 分发：dist/runtime 五文件包 + SHA256 manifest 哈希链（版本/工具链/脏树指纹），三语装载同合同校验

### 绑定

- **C/C++**：include/shape_matching.h 完整语义合同（错误分层/所有权/并发/坐标与变换约定）
- **Python**：shape_sdk 包（ctypes 运行时加载），pyproject 环境契约（numpy>=2,<3；benchmark extra 含 opencv-python）
- **Rust**：shape-sdk crate（libloading 运行时加载，unsafe 收敛于 ffi 模块），四帧事实基准与 Python 同机逐帧对齐

### 质量

- 深度审计-修复-复审全环闭合（2026-11-17）：14 项原始发现+循环新增 4 项全部实证修复，复审四态判定在册
- 门禁：ctest 7 套（组件/ABI 链/篡改回归/口径见证）+ Python 主链 7 组（构建管线强制）+ Rust 17 项
- 跨语言一致性：四帧事实基准 Python/Rust 同机对齐（中心偏差 0.05-0.14px，同分同角，无锚帧零误报）

### 兼容与迁移

- 旧默认参容器（meta.cfg 三参为零，v0.1.0 早期构建产物）加载被拒：用 `scripts/migrate_zero_cfg.py` 从模型段恢复实际参数，或以原图重训——决策树见 README「兼容性与迁移」
- ABI 版本 3 与产品版本相互独立；消费方凭 manifest 的 sdk_revision/anchor 检测上游更新

### 已知边界（如实，非语义风险）

- 清洁环境完全可重复构建未验证（工具链版本已记录于 manifest）
- 真机/现场图像、长期压力与并发极限未验收
- Python 依赖支持矩阵（Python 3.9 最低版/NumPy 小版本范围）未全验
- 构建脚本以 Windows x64 + MSVC + OpenCV world 为前提，其他平台未承诺
