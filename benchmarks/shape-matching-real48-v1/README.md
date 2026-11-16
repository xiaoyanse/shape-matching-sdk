# 形状匹配原48例可恢复材料 v1

这组材料用于恢复之前的三原图、48个已知变换案例。它不是用模型灰度模板贴到空白背景的替代图。

## 已追溯来源

- 原生成器：提交 `802a7ba4` 的 `scripts/benchmark/shape_match_vs_halcon.py`，Git blob `3ddcbb3e34a7e6b39dadfb1262f71ee2fedebe46`。完整原字节保留为 `generator.source.txt`；重建仅执行其 `cmd_gen`，不运行其他引擎入口。
- 原始输入：046-halcon 项目上传图库按PNG文件名排序的前三张，均为5472×3648。随图原元数据记录名称分别为 `tts-camera-1792142044098.png`、`tts-camera-1792137079285.png`、`large-frame-roll-90-f12.png`。这些名称证明上传记录，不能据此认定为HALCON官方示例或已验证的现场实拍。
- 模型：2026-10-25性能验证保留的 `model-img0.bin` 至 `model-img2.bin`，全部SHA-256与历史 `final-comparison.json` 相符。
- 缩放后均为4096×2730，原梯度选窗结果依次为 `[950,1700,500,500]`、`[1450,1700,500,500]`、`[700,1950,500,500]`。三个模型内嵌PNG解码后与对应ROI灰度逐像素完全相同。
- `historical-evidence.json` 保留历史报告的模型指纹、ROI、48例名称和结果/误差摘要，以及两份历史来源报告的整文件指纹。不将历史耗时视为当前运行结果。

`fixture-manifest.json` 冻结原图、元数据、模型、原生成器及历史摘要指纹，也保留本次确定性重建所得51张图的解码BGR像素指纹（3参考图＋48变换图）。

原图和模型保留原字节。上传元数据以等价JSON缩进及末尾换行保存，原始元数据文件字节指纹另记在各引用的 `uploaded_metadata_origin_sha256`，不冒称其排版字节未变。

## 真值与字节身份

原生成器将原图缩至最长边4096，以500×500梯度密度选窗。每图包含4平移、8旋转、2亮度和2整数均匀噪声案例，随机种子按原配方固定。噪声不是高斯噪声；这48例也不包含独立的双目标专项验收。

原基准目录 `manifest.json` 的历史SHA-256为：

`6700b84c35c3b2db22b619e917b5bde393be7898ee1d86e52fc1ce5bb63d835d`

按原绝对路径、JSON indent=1、Windows CRLF、无末尾换行重建可精确复现这一整文件指纹。这确认名称、顺序、ROI和所有真值；校验还逐案重算历史结果的位置/角度误差。

迁址后的manifest绝对路径不同，整文件指纹自然不同。工具同时核对路径规范化后的完整内容并分别报告两种SHA。旧衍生PNG的文件/像素指纹未留存，不能声称已逐文件证明与消失的旧PNG字节完全相同；现有证据为相同原始输入、原生成器字节、精确历史manifest及模型ROI像素。后续重建必须匹配本次冻结的全部解码像素指纹；编码器变化导致PNG压缩字节不同不影响像素身份。

## 使用

在项目根目录，以项目虚拟环境运行。工具不加载DLL、不训练、不导入HALCON、不读取业务库或启动服务。

```powershell
apps/business-server/python/.venv/Scripts/python.exe scripts/benchmark/preserve_shape_benchmark.py verify

apps/business-server/python/.venv/Scripts/python.exe scripts/benchmark/preserve_shape_benchmark.py rebuild --output build/verification/shape-real48-v1

apps/business-server/python/.venv/Scripts/python.exe scripts/benchmark/preserve_shape_benchmark.py verify-generated --output build/verification/shape-real48-v1
```

输出目录必须是 `build/verification` 下的全新目录，拒绝覆盖已有目录、跨根或经过目录链接。失败后不自动删除材料或残留目录；先核清本轮生成物再按项目清理边界回收。

完整衍生图约262 MiB，只在build中生成、可重建，不提交48张大图。三张原图与三个模型合计约17 MiB，保存于项目本机 `data/shape-matching-real48-v1/` 的 `sources/` 和 `models/`，不上传 Git。运行以上命令前，将该目录的六个文件复制到本目录对应的 `sources/` 和 `models/`，再运行 `verify`；校验依照清单逐项核对 SHA-256。

**素材仅保存在本机，没有远端备份。** Git 克隆不包含六个文件；换机前须另行复制并按清单验证。仓库中的清单、配方和真值不等于素材备份。

源图、模型和配方属于长期基准材料，不纳入build/临时测试清理。正式48例精度/性能回归、双目标专项及现场验收仍需另行执行；本包只提供材料恢复与一致性验证。
