#pragma once
// 引擎面粗层 GPU 服务（W2-B）：VF_SHAPE_BACKEND 三态后端 + GPU 粗层匹配编排
//（上下文生命周期/模型驻留缓存/帧路径/宿主提取收尾）。服务层失败=编码错误直报
// 不静默回退（auto 态回退兜底在 match_two_stage 接缝层，W4）；进程级单上下文+
// 互斥串行。设计见 w2b-engine-seam-brief §2 与 w4-fallback-brief §1。
#include "../model.hpp"

#include "gpu_device.hpp"

#include <string>
#include <vector>

namespace vfshape::gpu {

// 后端三态（W4）：VF_SHAPE_BACKEND=cpu（强制，不触 GPU）| gpu（强制 opt-in，
// 失败直报）| auto（默认，探测+运行中失败回退）；current_backend 每调用读 env
//（进程内可翻转），resolve_backend 对 auto 折叠探测与粘性禁用，供 match_two_stage
// 入口一次决策整调用同后端
enum class Backend { Cpu, Gpu, Auto };
Backend current_backend();
Backend resolve_backend();

// 运行中失败连续计数（粗精共享，进程级）：成功清零；连续 3 次失败转粘性禁用
//（日志一行），此后 auto 态直取 CPU
void note_gpu_call_ok();
void note_gpu_call_failed();

// 最近一次 GPU 粗层调用的单列证据（链路/提取耗时、initial 候选数）
struct CoarseServiceStats {
    double gpu_chain_ms = 0;
    double extract_ms = 0;
    int initial_count = 0;
};
const CoarseServiceStats& last_call_stats();

// match(model, small, threshold) 的 GPU 等价（two_stage 粗层段语义，候选收尾同链）
std::vector<vf_shape_candidate> match_coarse_gpu(Model& coarse, const cv::Mat& coarse_img,
                                                 double threshold);

}  // namespace vfshape::gpu
