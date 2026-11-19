# GPU 可行性尖峰：匹配热路径最小链路 CUDA 移植实测

目的：用最小链路（梯度金字塔 / spread / 响应累加三核 + 两级编排）实测自写 CUDA 核
相对 CPU 参考的加速比，为全量移植决策供数。**不修改引擎任何文件**——模板数据经
官方 Python 绑定训练导出的 VF2S3 容器字节由本目录独立解码，相似度 LUT 由脚本从
vendor 源逐字提取生成头文件。

## 目录

- `src/common.h` 共享结构 + LUT 查表 + 候选提取/精修（CPU/GPU 同一份，保证对拍差异只来自内核）
- `src/cpu_chain.{h,cpp}` CPU 单线程参考链（GaussianBlur/pyrDown/resize 用 OpenCV 同参调用，其余标量直译 vendor 语义）
- `src/gpu_chain.{h,cu}` CUDA 三核与两级链（特征常驻显存，精修窗批量单内核）
- `src/model_source.{h,cpp}` VF2S3 容器解码 + 帧读取/缩放/补边/裁窗
- `src/spike_main.cpp` CLI：对拍门禁 + 基准
- `scripts/dump_model.py` 数据管道：fact 帧 1 训练锚模板（±30°/5° 单尺度，同 run_facts.py spec）
- `scripts/gen_lut.py` 从 vendor line2Dup.cpp 提取 SIMILARITY_LUT 生成 `src/similarity_lut.h`
- `scripts/pe_imports.py` PE 导入预验（弹窗红线：启动前核对 DLL 可解析）
- `scripts/engine_frame_time.py` 引擎全帧耗时基线（生产 CPU 实现对照）
- `build.bat` nvcc 三代 fatbin 构建（sm_86/89/120）

产物在 `build/gpu-spike/`（gitignore 区）：`gpu_spike.exe`、`data/anchor-model.bin`、
`run-1080p.log`、`run-native.log`。

## 复现步骤

```bash
# 1) 数据管道：训练锚模板并落盘（约 2 分钟）
SHAPE_SDK_RUNTIME=dist/runtime uv run --project bindings/python \
    python benchmarks/gpu-spike/scripts/dump_model.py
#    （similarity_lut.h 已随仓；vendor 表变更时重跑 scripts/gen_lut.py）

# 2) 构建（vcvars 完整环境 + nvcc 三代 gencode，约 1-2 分钟）
cmd //c "benchmarks\\gpu-spike\\build.bat"

# 3) 启动前导入预验（PATH 前置 OpenCV bin）
PATH="/c/opencv/build/x64/vc16/bin:$PATH" \
    python benchmarks/gpu-spike/scripts/pe_imports.py build/gpu-spike/gpu_spike.exe

# 4) 对拍 + 基准（默认 1920x1080，阈值 70，预热 10 × 100 次取中位）
PATH="/c/opencv/build/x64/vc16/bin:$PATH" ./build/gpu-spike/gpu_spike.exe
#    原生分辨率：--width 1280 --height 960；仅对拍 --mode correctness；仅基准 --mode bench

# 5) 引擎基线对照
SHAPE_SDK_RUNTIME=dist/runtime uv run --project bindings/python \
    python benchmarks/gpu-spike/scripts/engine_frame_time.py
```

退出码：0=门禁通过；2=对拍门禁失败；3=运行错误。异常路径全 try/catch 退出码化，
`SetErrorMode` 关闭系统错误弹窗。

## 内核与 vendor 语义对照

| 尖峰内核 | vendor 对应 | 语义要点 |
| --- | --- | --- |
| k_blur_h/v | `GaussianBlur` 7×7 BORDER_REPLICATE | 定点核 /256={8,28,56,72,56,28,8}；中间不舍入、末轮单次舍入——实测与 OpenCV 逐字节一致 |
| k_pyrdown | `cv::pyrDown` | 5 抔 {1,4,6,4,1}/16，REFLECT_101 边界，末轮单次舍入——实测逐字节一致 |
| k_grad_quant | `strongestGradient` + `convertTo(CV_8U,16/360)` | 三通道 Sobel 取范数最大通道；rint(round-half-even)×16/360 后 &7；边界像素桶=0 |
| k_hysteresis | `hysteresisGradient` | 平方幅度 > weak² 处 3×3 直方图（边界零计票），首最大桶 ≥5 票成 one-hot |
| k_spread_h/v | `spread`/`orUnaligned8u` | 因果 T 窗（右/下方向）按位 OR，可分离两程 |
| k_sim_full | `similarity` + `linearizeResponses` | R(label,mask)=max(LUT[32l+lo],LUT[32l+16+hi]) 逐位同表；出界特征跳过（见边界说明） |
| k_sim_windows | `similarityLocal` | 16×16 窗（T=4），中心夹取/行主序首最大/B5 抛物线亚像素在共享代码 |

## 已知边界（诚实清单）

- 参考链与 GPU 用精确 `atan2`；引擎 `cv::phase` 走 fastAtan2（~0.3° 近似）。首轮
  对拍（CPU 侧用 cv::phase 时）实测方向桶一致率 99.05–99.55%，链顶候选不受影响
  （同位姿、偏差 ≤0.13px）。生产化可选复刻 fastAtan2 或接受该差。
- 响应累加对出界特征直接跳过；vendor 线性内存 flat 越界会行回卷，仅影响右/下边缘
  窄带（顶部候选不受影响，锚居中）。全图打分范围 W×H（vendor 只写 span 内并回卷）。
- 精修窗批量后 GPU 阶段时间常显示 0.000ms（内核 <1μs 量级），宿主侧候选提取
  0.1–0.4ms 单列（系统负载下可至 1.5ms，含传输口径波动主要来自此处）。
- fatbin 只嵌三代 SASS 无 PTX：跨更代架构（>sm_120）需加 `code=compute_XXX` PTX
  兜底或重编。
- 基准为同帧重复（确定性）；1080P 帧为原生 1280×960 双线性放大 1.5×，锚超出
  训练尺度、引擎空结果属预期（内核计时不受影响；引擎基线用原生帧）。
- 链 B 精修窗只跑 1 窗（生产最多 8 窗按需调度）；多窗批量/流水、CUDA Graph、
  训练侧并行未在尖峰范围。

## 环境坑（本机实测）

- nvcc 前端按系统代码页（GBK）词法分析 `.cu` 字符串字面量，`-Xcompiler "/utf-8"`
  只救宿主编译——`.cu` 内字符串字面量须 ASCII（中文仅注释）。
- 链接 OpenCV 用 `-L"..."`（`-LIBPATH:` 不被 nvcc 转发）。
- 批处理文件含 UTF-8 中文会被 cmd 按 GBK 误析断行——批处理保持纯 ASCII。
