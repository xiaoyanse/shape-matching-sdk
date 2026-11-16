# shape-matching-sdk

形状模板匹配引擎 SDK：C++17 动态库（C ABI v3）+ 官方 Python ctypes 绑定。
提供模板训练、加载、导出与单/多模板旋转缩放匹配，面向工业视觉定位场景。

## 交付形态

- 引擎：`visionforge_shape.dll`（Windows x64），训练/加载/导出/单模板与多模板匹配同库完成
- 算法底座：line2Dup（shape_based_matching，BSD-2）+ 自研两段式匹配与几何精修，见 `THIRD_PARTY_NOTICES.md` / `UPSTREAM_PATCHES.md` / `vendor-integrity.json`
- 绑定：`bindings/python/shape_sdk/`（ctypes，manifest 哈希校验 + ABI 协商内置）
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

## ABI 兼容合同（消费方对接必读）

- C ABI 出口固定宽度结构体，版本号 `vf_shape_abi_version()`，当前 **3**
- 兼容性纪律：新增字段/函数须升次版本且旧绑定协商继续生效；破坏性变更必须升主版本，
  消费方绑定随制品同步分发
- `runtime-manifest.json` 收录全部制品文件 SHA256 及 `sdk_revision`（构建时 git 提交号）
  与 `anchor`（最近 tag 描述）；换装以 manifest 校验为准，消费方不得单文件替换
- 建议发版走 git tag（semver）+ Release 制品包，消费方凭 manifest 的 revision 字段
  自动检测上游更新
- OpenCV 运行库随制品分发（当前 opencv_world 4.10），与消费方自持的其他 OpenCV 版本互不共享

## 目录结构

```
src/                  引擎源码（abi/matcher/two_stage/geometric_refinement/coarse_image/model_codec）
include/              C ABI 头文件
vendor/shape_based_matching/  上游 line2Dup + MIPP（字节冻结，见 vendor-integrity.json）
tests/                C++ 几何/帧准备回归测试（-BuildTests 启用）
bindings/python/      官方 Python 绑定包 shape_sdk
scripts/              构建与装配脚本
```
