#pragma once
// GPU 设备面内部共享定义（仅 src/gpu/*.cu 包含，不对中性头暴露）：两级帧工作面
// 缓冲布局、设备侧特征记录（int4）与不透明上下文实体。W2 模板尺寸驻留；W3 起
// 帧工作面/响应/窗分暂存/fine 特征槽按 frame_slot 双份（粗精交替与流水化双 job
// 并存的设备面隔离，默认槽 0 = W2 行为路径）。
#include <cstddef>
#include <cstdint>

namespace vfshape::gpu {

// 每级帧工作面（bgr=输入帧/中间帧，smooth=高斯/pyrDown 输出，bucket=16 桶量化，
// angle=迟滞 one-hot，spread=因果 T 窗 OR，mag=最强通道平方幅度）
struct LevelBuffers {
    uint8_t *bgr = nullptr, *smooth = nullptr;
    uint32_t* blur_tmp = nullptr;
    uint8_t *bucket = nullptr, *angle = nullptr, *spread = nullptr, *spread_h = nullptr;
    float* mag = nullptr;
    int w = 0, h = 0;
};

// 设备侧特征记录（尖峰 int4 布局：x/y/label/pad）
struct Feature4 {
    int32_t x = 0, y = 0, label = 0, pad = 0;
};

// 上下文实体（gpu_device.hpp 前置声明；创建/销毁在 gpu_device.cu）。
// frame_slot：0/1 双份工作面（流水化双 job 并存）；pyramid_level 沿 [级]。
struct GpuContext {
    LevelBuffers face[2][2];  // [frame_slot][pyramid_level]：[x][0]=L0 档，[x][1]=L1 档
    Feature4* feats = nullptr;   // coarse 模型特征（slot0，单份：串行粗层路径）
    int *feat_begin = nullptr, *feat_count = nullptr;
    int2* templ_dims = nullptr;
    Feature4* fine_feats[2] = {nullptr, nullptr};   // fine 窗特征（slot1）×frame_slot
    int* fine_begin[2] = {nullptr, nullptr};
    int* fine_count[2] = {nullptr, nullptr};
    int2* fine_dims[2] = {nullptr, nullptr};
    uint16_t* response[2] = {nullptr, nullptr};     // 每 pose 全图响应 ×frame_slot
    int4* centers[2] = {nullptr, nullptr};          // 窗分候选中心暂存 ×frame_slot
    uint16_t* window_scores[2] = {nullptr, nullptr};
    int candidate_capacity = 0;
    size_t response_capacity = 0;
    int pose_capacity = 0;
    int feat_capacity = 0;
    int resident_poses = 0;
    int resident_feats = 0;
    bool features_ready = false;
    bool dims_ready = false;
    int fine_resident_poses[2] = {0, 0};
    int fine_resident_feats[2] = {0, 0};
    bool fine_features_ready[2] = {false, false};
    bool fine_dims_ready[2] = {false, false};
    int cur_w[2] = {0, 0}, cur_h[2] = {0, 0};  // 各 frame_slot 容量上传后的当前帧档
};

}  // namespace vfshape::gpu
