#ifndef GPU_SPIKE_GPU_CHAIN_H
#define GPU_SPIKE_GPU_CHAIN_H
// CUDA 链：三核（梯度金字塔/spread/响应累加）+ 两级编排，与 CPU 参考同语义。
#include "common.h"

namespace gpuspike {

struct GpuChainContext;

// 创建/销毁（缓冲按容量一次性分配，基准循环内复用）
GpuChainContext* gpu_create_context(int max_w, int max_h, int max_poses, int max_feats,
                                    int max_candidates, int max_positions, std::string& err);
void gpu_destroy_context(GpuChainContext* ctx);

// 特征一次性驻留：L1/L0 特征拼入同一缓冲，begin 表前半=L1、后半=L0
bool gpu_upload_features(GpuChainContext* ctx, const std::vector<PoseModel>& poses,
                         std::string& err);

// 单次链路。upload=false 时跳过 H2D（帧已驻留，供"不含传输"口径）。
// correctness=true 时把 angle/spread 中间图回填到 out.maps（D2H 仅对拍用）。
bool gpu_run_chain(GpuChainContext* ctx, const uint8_t* padded_bgr, int w, int h, float weak,
                   const std::vector<PoseModel>& poses,
                   float coarse_th, float fine_th, bool upload, bool correctness,
                   ChainOutput& out, std::string& err);

// 探测设备名与算力（编译目标验证用）
bool gpu_device_info(std::string& name, int& major, int& minor, std::string& err);

}  // namespace gpuspike

#endif
