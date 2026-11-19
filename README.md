# shape-matching-sdk

形状模板匹配引擎 SDK：C++17 动态库（C ABI v3）+ 官方 Python ctypes 绑定 + 官方 Rust 绑定。
提供模板训练、加载、导出与单/多模板旋转缩放匹配，面向工业视觉定位场景。

## 交付形态

- 引擎：`visionforge_shape.dll`（Windows x64），训练/加载/导出/单模板与多模板匹配同库完成
- 算法底座：line2Dup（shape_based_matching，BSD-2）+ 自研两段式匹配与几何精修，见 `THIRD_PARTY_NOTICES.md` / `UPSTREAM_PATCHES.md` / `vendor-integrity.json`
- 绑定：`bindings/python/shape_sdk/`（ctypes）与 `bindings/rust/shape-sdk`（libloading 运行时装载），均内置 manifest 哈希校验 + ABI 协商
- 制品目录：`dist/runtime/`（DLL + OpenCV 运行库 + 许可证 + `runtime-manifest.json`，构建后生成，不入库）

## 构建

前置：MSVC C++ 工具链（vcvars64 可发现）、OpenCV 4 C++ SDK（默认 `C:\opencv\build\x64\vc16`，可被参数覆盖）、CMake、Ninja、PowerShell 7 或 Windows PowerShell。

```powershell
# 编译并装配制品到 dist/runtime（含上游 vendor 字节校验）
scripts/build_native.ps1

# 附带 C++ 回归测试
scripts/build_native.ps1 -BuildTests

# 实验 AVX2 内核 / 诊断计时（不装配，不替换通用运行库）
scripts/build_native.ps1 -Avx2 -SkipAssemble
```

构建脚本自动发现 vswhere 最新 MSVC；也可显式传 `-VcVarsPath` / `-OpenCvRoot`。

## Python 绑定使用

```python
import os
os.environ["SHAPE_SDK_RUNTIME"] = r"<制品目录>"   # 缺省为仓内 dist/runtime
from shape_sdk import train_model, NativeModel, match_batch
```

绑定加载时逐文件校验 manifest SHA256 并协商 ABI 版本（当前 3）；不匹配即拒绝加载。

## Rust 绑定使用

`bindings/rust/shape-sdk`（crate 名 `shape-sdk`，edition 2024）：

```rust
use shape_sdk::{train_model, BgrImage, Model, TrainParams};

// BgrImage 合同：连续 uint8 BGR，data.len() == 3×宽×高，宽高 24–4096
let reference = BgrImage::new(1280, 960, &bgr_buffer)?;
let params = TrainParams {
    roi_xywh: (30, 30, 200, 200),
    angle_min: -30.0, angle_max: 30.0, angle_step: 5.0,
    scale_min: 1.0, scale_max: 1.0, scale_step: 0.05,
    detector: None,                        // None=原生默认三参 (128/30/60)
};
let model_bytes = train_model(reference, &params)?;
let model = Model::from_bytes(&model_bytes)?;
let hits = model.match_(reference, 70.0, 16)?;   // 阈值 (0,100]，容量 2..256
model.close()?;                                   // 幂等；Drop 兜底
```

- 环境契约与 Python 侧一致：crate 不内嵌原生库，运行时目录优先取
  `SHAPE_SDK_RUNTIME` 环境变量（制品目录），缺省仓内 `dist/runtime`（按编译期
  crate 位置上溯仓根，前提是 crate 位于仓内 bindings/rust/shape-sdk；外部集成
  须设环境变量）。装载即校验 manifest 全文件 SHA256 并协商 ABI 版本 3
- MSRV 1.85；运行依赖 libloading/serde/serde_json/sha2，测试另依赖 `image`
  （dev-only，PNG 解码基准帧）
- 与 Python 侧的关系：同一 C ABI v3 之上的平行安全层，训练/匹配/批量/元数据
  语义与错误口径对齐 `bindings/python/shape_sdk`。测试同样对齐：`tests/smoke.rs`
  + `tests/abi_chain.rs` 对应 `scripts/test_main_chain.py` 核心组（全链/元数据
  往返/detector 往返/批量混合/生命周期），`tests/polygon.rs` 对应
  `scripts/test_polygon_chain.py` 多边形组，`tests/conformance.rs` 复刻
  `benchmarks/run_facts.py` 四帧事实基准全语义（同一 spec 真值断言）

## 多边形模板能力（ABI v3 加性导出）

在矩形 ROI 之外支持任意多边形训练掩码（外环+洞），位姿/预算/导出/加载/匹配
全部复用矩形路径（外环外接矩形即 ROI），矩形训练零行为变化。

```python
from shape_sdk import train_polygon_model, suggest_contours

# 自动轮廓提取建议（Canny→外轮廓→多边形逼近，确定性；返回顶点集合）
suggestions = suggest_contours(reference_bgr)          # 或 roi_hint=(x, y, w, h)

# 训练：outline 3–64 顶点任意绕向（SDK 归一）；洞 ≤8、每洞面积 ≥64px²
# 且完全在外环内；角度/缩放/阈值语义与 train_model 同名参数一致
data = train_polygon_model(reference_bgr, outline, holes=[hole],
                           angle_min=-30, angle_max=30, angle_step=15,
                           scale_min=1.0, scale_max=1.0, scale_step=1.0)
model = NativeModel(data)
model.polygon_metadata   # {"outline": [...], "hole_count": 1}（矩形资产回退 ROI 四角、洞数 0）
```

Rust 绑定同构：`train_polygon_model` / `suggest_contours` / `Model::polygon_metadata`
（`PolygonTrainParams` / `PolygonVertex` / `PolygonMetadata`）。

- **几何合同**：顶点须落在参考图坐标域内；外环不得自交或退化；外接矩形满足
  24..4096 像素合同并作为姿态/画布/像素预算记账基（短边 <48px 时两级粗层无法
  满足最小 ROI 合同，与矩形训练同一行为）；洞语义=训练特征提取期掩蔽
  （洞内像素不产特征），洞间重叠按并集掩蔽不校验
- **错误码**：几何非法返回 `VF_SHAPE_INVALID_POLYGON(17)`，params 为
  [细分类型, 主参数, 次参数]（细分类型含义见头文件枚举注释）
- **`vf_shape_suggest_contours` 返回值语义特殊**：≥0 为建议条数（0=无建议，
  非错误），-1=可报告失败，-2=未分类原生失败
- **容器兼容承诺**：多边形资产容器 meta 段携带可选 `poly` 键（归一化外环顶点+
  洞分段+外接矩形）；**旧版本库加载多边形容器=合法**（忽略 poly 键、按外接矩形
  回退解读，几何回读为外接矩形近似）；新版本库加载旧容器（无 poly 键）=矩形
  语义不变；poly 键几何非法或与 cfg.roi 不一致按 MODEL_TAMPER/INVALID_POLYGON 拒绝
- **耗时承诺**：1080P 参考图 + 35 位姿、外接矩形 ≤400px 的多边形训练 ≤2s
  （实测基线见 tests/polygon_test.cpp 耗时背书用例）

## ABI 兼容合同（消费方对接必读）

- C ABI 出口固定宽度结构体，版本号 `vf_shape_abi_version()`，当前 **3**
- 兼容性纪律：新增字段/函数须升次版本且旧绑定协商继续生效；破坏性变更必须升主版本，
  消费方绑定随制品同步分发
- `runtime-manifest.json` 收录全部制品文件 SHA256 及 `sdk_revision`（构建时 git 提交号）
  与 `anchor`（最近 tag 描述）；换装以 manifest 校验为准，消费方不得单文件替换
- 建议发版走 git tag（semver）+ Release 制品包，消费方凭 manifest 的 revision 字段
  自动检测上游更新
- OpenCV 运行库随制品分发（当前 opencv_world 4.10），与消费方自持的其他 OpenCV 版本互不共享

## 兼容性与迁移

0.1.0 起持久化校验收紧：detector 三参在持久化侧必须为正值。84ec8d9（P1）及更早版本
用默认训练参数生成的两级容器，其 `meta.cfg` 三参写为 `[0,0,0]`，当前版本加载将拒绝
（错误 `persisted detector thresholds must be positive`）。单级旧格式资产不受此收紧
影响（其段内始终序列化实际三参）。

更早方言（84ec8d9 之前，两级容器 meta 段无 `ref`/`cfg` 键）的加载受容为**合同行为**：
按派生回退语义成功加载（参考尺寸派生为模板灰图尺寸、角度步派生为 fine 组细分步、
三参取段内序列化实际值），未来版本不会收紧拒绝。`migrate_zero_cfg.py` 对该方言
幂等直通（原样输出，无需迁移）。

迁移决策树：

1. **有原始参考图与训练配置** → 用当前版本重新训练并导出，最干净；
2. **无原图** → 用迁移工具从容器内各 fine 段恢复实际三参并回填 `meta.cfg`：
   ```
   python scripts/migrate_zero_cfg.py <旧容器> <新容器>
   ```
   工具校验各 fine 段序列化三参一致后改写元数据，并用当前版本加载核对 metadata
   三参；段间不一致或三参已非零时原样输出（幂等）；
3. **保留旧版本库** → 只读消费旧资产（容器 magic 与 ABI 未变，旧库可继续读取其
   自产资产）。

**不要**通过删除 `ref`/`cfg` 元数据规避校验：参考尺寸与训练配置会永久丢失，容器
退化为弱回退解读（参考图几何仅剩派生值），不可逆。

## 基准与回归

- `benchmarks/run_facts.py`：**端到端确定性基准**（官方绑定 + shape-facts v1 合成事实帧）——帧 1 训练锚模板（±30°/5°、单尺度），对四帧匹配并对照 spec 预期真值（在场性 + 中心定位 ≤8px），输出训练/逐帧耗时。当前基线：训练 ~0.6s（13 位姿）、匹配 5–10ms/帧（1280×960）、四帧全过（中心偏差 <0.2px、零误报）
- `benchmarks/shape-facts-v1/`：合成基准资产（4 帧确定性 PNG + 图案生成器源码 + 机器可读规格含预期真值与字节哈希）
- `benchmarks/shape-matching-real48-v1/`：48 例真实素材基准（4 平移×8 旋转×2 亮度×2 噪声，含历史结果真值指纹与生成器源档）
- `tests/`：C++ 单元回归（几何精修/帧准备/粗图/聚类/多边形训练与洞掩蔽，`-BuildTests` 启用）
- `scripts/test_main_chain.py`：Python 绑定主链回归（训练→元数据→往返→容器篡改→批量→生命周期），
  `-BuildTests` 在装配完成后自动执行，为构建门禁一部分（需 python + numpy）
- `scripts/test_polygon_chain.py`：Python 绑定多边形合同回归（多边形训练→元数据
  往返→容器 poly 方言→suggest_contours，需 python + numpy）
- `scripts/smoke_binding.py`：绑定加载冒烟（manifest 校验 + ABI 协商）
- `bindings/rust/shape-sdk/tests/`：Rust 绑定回归（smoke + abi_chain 五组核心断言，
  对齐 `scripts/test_main_chain.py` 与 `tests/abi_chain_test.cpp`）+ `conformance.rs`
  四帧事实基准（复刻 `run_facts.py` 全语义，Rust 实测与 Python 基线对齐：
  中心偏差 <0.2px、零误报）

引擎改动后跑 `run_facts.py` 与 CTest 作为最低回归门；性能对比基准（多线程批量、
与第三方引擎对拍）由消费方按需在其环境构建。

## 绑定路线图

C ABI 单合同 + 官方绑定按真实消费方滚动补齐（不做全家桶）：

| 语言 | 状态 | 说明 |
| --- | --- | --- |
| C/C++ | 已支持 | `include/shape_matching.h` 头文件即绑定；最小接入见 `examples/README.md` |
| Python | 已支持 | `bindings/python/shape_sdk`（ctypes，manifest 校验内置） |
| Rust | 已支持 | `bindings/rust/shape-sdk`（libloading 运行时装载，manifest 校验内置，MSRV 1.85） |
| C# | 按需 | P/Invoke，消费方出现时补官方包 |
| Java/Kotlin | 按需 | JNA 优先、JNI 备选；目标平台（JVM/Android）明确后实施 |

非承诺范围：Linux/macOS/移动端二进制与交叉编译不在当前 Windows x64 包内，
需要时按平台扩展工程单独立项。

## 目录结构

```
src/                  引擎源码（abi/matcher/two_stage/geometric_refinement/coarse_image/model_codec）
include/              C ABI 头文件
vendor/shape_based_matching/  上游 line2Dup + MIPP（字节冻结，见 vendor-integrity.json）
tests/                C++ 几何/帧准备回归测试（-BuildTests 启用）
bindings/python/      官方 Python 绑定包 shape_sdk
bindings/rust/        官方 Rust 绑定 crate shape-sdk
scripts/              构建与装配脚本
```
