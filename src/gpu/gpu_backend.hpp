#pragma once
// GPU 后端骨架（W1，规划 .artifacts/proposals/gpu-port-phase0.md）：
// 设备上下文（容量制缓冲）+ 从内存 Model 提特征驻留 + 帧 H2D。
// 本期不接匹配路径——prepare/match 面的后端开关与内核在 W2/W3 引入。
// 接口 CUDA 中立：设备面见 gpu_device.hpp（实现 gpu_device.cu）；
// 本头补 Model 级驻留适配（实现 gpu_backend.cpp，遍历引擎 Detector 模板）。
// 仅 VF_SHAPE_CUDA=ON 时参与编译；OFF 构建不引用本头，产物与纯 CPU 现状等价。
// 加载方案（规划 §二拍板·甲）：单 DLL + 静态 cudart，运行时探测。
#include "../model.hpp"

#include "gpu_device.hpp"

namespace vfshape::gpu {

// 驻留统计：一致性断言口径（与引擎 Detector 模板逐项对账）
struct ResidencyStats {
    int pose_count = 0;       // 驻留位姿数（= Model::poses.size()）
    int features_level0 = 0;  // L0(T=4) 层特征总数
    int features_level1 = 0;  // L1(T=8) 层特征总数
    int feature_total = 0;    // 两层合计（设备 feats 缓冲占用量）
};

// 从内存 Model 提特征驻留：读引擎 Detector 模板（getTemplates），不走容器
// 解码路径。设备布局沿尖峰——feat_begin/feat_count 前半 = L1（slot=id）、
// 后半 = L0（slot=pose_capacity+id），feats 单缓冲先 L1 后 L0 拼接。
bool upload_model_features(GpuContext* ctx, const Model& model, std::string& err,
                           ResidencyStats* stats = nullptr);

}  // namespace vfshape::gpu
