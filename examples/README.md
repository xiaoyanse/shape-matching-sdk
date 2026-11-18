# examples —— 三语接入样例

SDK v0.1.0 下游接入的最小可运行样例与 C/C++/Python/Rust 指引。唯一契约是
C ABI v3（`include/shape_matching.h`，8 个导出入口）；Python / Rust 官方绑定
是其上的平行安全层，训练/匹配/批量/元数据语义与错误口径一致。

| 文件 | 说明 |
| --- | --- |
| `python/train_match_batch.py` | Python 全链样例：合成图训练 → 元数据 → 单帧匹配 → 批量匹配 → 生命周期 |

## 通用前提（三语一致）

1. **制品**：`scripts/build_native.ps1` 产出 `dist/runtime/`（`visionforge_shape.dll` +
   `opencv_world4100.dll` + 第三方许可证 + `runtime-manifest.json`）。整目录分发与装载，
   装载方按 manifest 逐文件 SHA256 校验，**不得单文件替换**。
2. **运行时定位**（三语绑定同契约）：优先 `SHAPE_SDK_RUNTIME` 环境变量指向制品目录；
   缺省仅当绑定位于仓内时自动发现仓内 `dist/runtime`。消费方部署/打包必须显式设置
   环境变量，或严格按各语言缺省规则放置制品。
3. **图像合同**（train/match 输入统一）：连续 uint8 BGR 三通道，stride == 3×width，
   宽高 24–4096 像素，坐标原点左上。
4. **数值口径**：score 为 0–100 的特征方向相似度（阈值过滤用，非校准概率）；
   threshold ∈ (0,100]；单模型候选容量 2–256；批量 1–16 个模型；活跃句柄上限 128；
   `close` 对未知/已关闭句柄幂等。

## Python 接入（`bindings/python/shape_sdk`，ctypes）

**最小五步**：

1. **取包**：开发环境 `cd bindings/python && uv sync`；分发时整包安装
   `pip install bindings/python`，或直接把 `shape_sdk/` 包目录随制品携带。
   依赖 Python ≥3.9、`numpy>=2,<3`，无其他第三方依赖。
2. **设环境**：`import shape_sdk` 前设 `SHAPE_SDK_RUNTIME`（缺省为仓内 `dist/runtime`）。
   PyInstaller 冻结包可把制品放入 bundle 的 `shape_matching/` 子目录，免设环境变量。
3. **建图**：`np.ascontiguousarray(...)` 保证连续 uint8 BGR。
4. **训练/加载**：`train_model(reference, (x, y, w, h), angle_*, scale_*)` 得模型字节
   （两级容器）；`NativeModel(bytes)` 加载，支持 `with` 块与显式 `close()`（幂等）。
5. **匹配/批量**：`model.match(frame, threshold=60, max_candidates=16)` 得候选列表；
   `match_batch([(m1, 60, 16), (m2, 70, 8)], frame)` 得逐项结果
   （`list[候选] | NativeShapeError`，单项失败不影响其余项）。

**运行样例**：

```
cd bindings/python && uv sync
uv run python ../../examples/python/train_match_batch.py
```

候选字段：`score / pose_id / pose_ambiguous / center / angle / scale / corners / bbox /
transform`（2×3 行优先仿射，参考图整幅坐标 → 帧坐标）。metadata 回显训练配置
（detector 三参为物化值，0 仍表示原生默认 128/30/60 语义）。

## C/C++ 接入（`include/shape_matching.h` 即合同）

**最小五步**：

1. **取包**：头文件 `include/shape_matching.h` + 制品目录。dist 不带导入库：
   **LoadLibrary + GetProcAddress** 开箱即用（与官方绑定同款运行时装载）；要走
   显式链接需先用 `dumpbin /exports` + `lib /def:` 自建导入库。
2. **装载与版本协商**：`LoadLibrary` 指向制品目录内 `visionforge_shape.dll`
   （`opencv_world4100.dll` 与主库同目录即可被找到，否则提前 `SetDllDirectory` /
   加 PATH）。任何其他调用之前先取 `vf_shape_abi_version()`，非 3 立即停用。
3. **建图**：连续 BGR 8bit 缓冲，`stride == 3×width`，宽高 24..4096。
4. **训练/加载**：`vf_shape_train_config` 先置 `struct_size = sizeof(...)`，角度
   [-180,180] step>0、缩放 [0.25,2] step>0、`weak_th ≤ strong_th ≤ 1020`、
   `feature_count` 16..512（三者置 0 = 默认 128/30/60）。**返回 0 才允许消费
   `*handle`**；`vf_shape_load` 直接吃模型字节（两级容器优先，旧单级格式兼容）；
   `vf_shape_export` 为两段式（先 `bytes=NULL, capacity=0` 查 `*size`）。
5. **匹配/批量**：`vf_shape_match(...)`；`vf_shape_match_batch(...)` 先检查整体返回码
   （为 0 才代表批量流程完成），再逐项检查 `items[i].status`；用完 `vf_shape_close`
   （幂等）。错误分层统一：返回 0 成功 / 1 可报告失败 / 2 未分类原生失败；
   error 缓冲由调用方提供（容量截断、保证 NUL 结尾）。

**最小骨架**（LoadLibrary 路线）：

```c
#include "shape_matching.h"
/* LoadLibrary("…\\dist\\runtime\\visionforge_shape.dll") 后逐个 GetProcAddress：
 *   vf_shape_abi_version                       → 必须 == 3
 *   vf_shape_train(image, size, w, h, stride, &cfg, &handle, err, cap)
 *   vf_shape_match(handle, image, size, w, h, stride, threshold,
 *                  candidates, capacity, &count, err, cap)
 *   vf_shape_close(handle, err, cap)           → 幂等
 * cfg.struct_size 必须先置 sizeof(vf_shape_train_config)；
 * 候选缓冲 candidates 由调用方分配（capacity 个 vf_shape_candidate）。
 * 同一句柄上的调用由库串行化；不同句柄可并行；匹配期间不得改写图像缓冲。 */
```

并发与寿命安全细节（并发 close、错误文本边界）以头文件注释为准，此处不重复。

## Rust 接入（`bindings/rust/shape-sdk`，crate 名 `shape-sdk`）

**最小五步**：

1. **取包**：`shape-sdk = { path = "<仓库>/bindings/rust/shape-sdk" }`（v0.1.0，
   edition 2024，MSRV 1.85；当前未发布 registry，走 path 依赖或随制品内嵌源码）。
   运行依赖 libloading/serde/serde_json/sha2。
2. **设环境**：`SHAPE_SDK_RUNTIME` 指向制品目录。**外部集成必须显式设置**——缺省
   发现只在 crate 位于仓内 `bindings/rust/shape-sdk` 时按编译期路径上溯仓根成立。
   装载即校验 manifest 哈希链并协商 ABI 3。
3. **建图**：`BgrImage::new(width, height, &bgr_bytes)?`（`data.len() == 3×宽×高`）。
4. **训练/加载**：`train_model(&img, &TrainParams { roi_xywh, angle_*, scale_*,
   detector: None, })?` 得 `Vec<u8>`；`Model::from_bytes(&bytes)?` 加载
   （`detector: None` = 原生默认三参 128/30/60）。
5. **匹配/批量**：`model.match_(img, 70.0, 16)?`；`match_batch(&[BatchRequest
   { model: &m, threshold, max_candidates }, …], img)?` —— 外层 `Err` 为整批失败
   （公共图像校验/编排级），外层 `Ok` 内逐项 `Result<Vec<Candidate>, ShapeError>`；
   `model.close()?` 幂等，`Drop` 兜底。

**活样例**：`bindings/rust/shape-sdk/tests/conformance.rs` —— 四帧事实基准全语义
（帧 1 裁锚邻域 260×260 训练 ±30°/5°，四帧按 spec 阈值匹配，对照真值判定在场性
与中心偏差 ≤8px），即"训练→匹配→判定"的完整真实用法；`tests/smoke.rs` +
`tests/abi_chain.rs` 覆盖主链合同（全链/元数据往返/detector 往返/批量混合/生命周期）。

## 下游打包注意

- **随制品分发**：绑定包（或 crate 源码）与 `dist/runtime` 整目录一起走，缺一不可；
  消费方凭 manifest 的 `sdk_revision` / `anchor` 检测上游更新。
- **ABI 升级纪律**：新增字段/函数升次版本且旧绑定协商继续生效；破坏性变更升主版本，
  绑定随制品同步分发——消费方锁定主版本号对接。
- **旧资产兼容**：0.1.0 起持久化校验收紧，早期默认参数容器的 `meta.cfg=[0,0,0]`
  加载被拒；无原图时用 `scripts/migrate_zero_cfg.py` 迁移，详见根 README「兼容性与迁移」。
- **不要**单文件替换 DLL、不要绕过 manifest 校验、不要删除容器内 `ref`/`cfg` 元数据。
