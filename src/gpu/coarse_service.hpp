#pragma once
// 引擎面粗层 GPU 服务（W2-B）：VF_SHAPE_BACKEND 后端开关 + GPU 粗层匹配编排
//（上下文生命周期/模型驻留缓存/帧路径/宿主提取收尾）。失败=编码错误直报不静默
// 回退（回退兜底属 W4）；进程级单上下文+互斥串行。设计见 w2b-engine-seam-brief §2。
#include "../model.hpp"

#include "gpu_device.hpp"

#include <string>
#include <vector>

namespace vfshape::gpu {

// 后端开关：VF_SHAPE_BACKEND=cpu（默认）| gpu（显式 opt-in）；每调用读 env，
// 测试进程内可翻转
enum class Backend { Cpu, Gpu };
Backend current_backend();

// 最近一次 GPU 粗层调用的单列证据（链路/提取耗时、initial 候选数）
struct CoarseServiceStats {
    double gpu_chain_ms = 0;
    double extract_ms = 0;
    int initial_count = 0;
};
const CoarseServiceStats& last_call_stats();

// match(model, small, threshold) 的 GPU 等价（two_stage 粗层段语义，候选收尾同链）
std::vector<vf_shape_candidate> match_coarse_gpu(Model& coarse, const cv::Mat& small,
                                                 double threshold);

}  // namespace vfshape::gpu
