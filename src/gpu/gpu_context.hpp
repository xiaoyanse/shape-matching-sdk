#pragma once
// GPU 设备面内部共享定义（仅 src/gpu/*.cu 包含，不对中性头暴露）：两级帧工作面
// 缓冲布局、设备侧特征记录（int4）与不透明上下文实体。W2 起新增模板尺寸驻留
// （templ_dims，vendor 相似度 span/回卷语义所需）与驻留就绪标志。
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

// 上下文实体（gpu_device.hpp 前置声明；创建/销毁在 gpu_device.cu）
struct GpuContext {
    LevelBuffers level[2];  // [0]=L0 全分辨率档，[1]=L1 半分辨率档
    Feature4* feats = nullptr;
    int *feat_begin = nullptr, *feat_count = nullptr;  // 前半 L1、后半 L0（尖峰布局）
    int2* templ_dims = nullptr;  // 每位姿每层模板 (width,height)，同 begin/count 布局
    uint16_t* response = nullptr;
    int4* centers = nullptr;
    uint16_t* window_scores = nullptr;
    size_t response_capacity = 0;
    int pose_capacity = 0;
    int feat_capacity = 0;
    int resident_poses = 0;
    int resident_feats = 0;
    bool features_ready = false;
    bool dims_ready = false;
};

}  // namespace vfshape::gpu
