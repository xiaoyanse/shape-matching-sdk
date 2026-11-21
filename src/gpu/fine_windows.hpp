#pragma once
// 精修窗批量评分 GPU 内核中立面（W3 波1）：16×16 窗分逐位复刻宿主
// accumulate_window（vendor similarityLocal 线性内存口径：出界特征跳过 + flat
// 行尾续读 + 相位回卷/Mat 尾取模确定化，语义基准 coarse_extract.cpp）。输出
// 布局 = 候选×256 uint16 行主序窗格（尖峰 refine_from_scores 消费口径）。
// 仅 VF_SHAPE_CUDA=ON 时有实现；OFF 构建不引用本头。
#include "gpu_device.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vfshape::gpu {

// 精修窗候选：clamp 后窗中心（L0 像素坐标；clamp 归调用方，vendor matchClass
// 同式 [8*T, size-templ.width-8*T]，见 extract_initial_candidates）
struct WindowCandidate {
    int x = 0, y = 0;
};

// 批量精修窗评分：候选数组 H2D 单次上传，窗分 D2H 单次回读 host_scores
//（cand_count*256）。feats_dev/feat_begin_dev/feat_count_dev/spread0_dev 为设备
// 驻留指针，传空 = 用 ctx 驻留（model_slot 选槽：0=coarse/W1 默认，1=fine 窗；
// feats 为 Feature4 设备打包布局，中立面不暴露内部结构故收 void*）；非空时
// 须与 ctx L0 档/驻留布局同形。帧档对 upload_frame_capacity 记录的当前档校验。
// L0 slot = pose_id_offset + cand_pose[i]
//（W1 布局传 context_pose_capacity）。ctx 提供 centers/window_scores 暂存与
// 容量校验；kernel_ms 非空回传内核事件耗时（memo 用）。
bool run_window_scores(GpuContext* ctx, const uint8_t* spread0_dev, int w0, int h0,
                       const void* feats_dev, const int* feat_begin_dev,
                       const int* feat_count_dev, int pose_id_offset,
                       const WindowCandidate* cands, const int* cand_pose, int cand_count,
                       std::vector<uint16_t>& host_scores, double* kernel_ms, std::string& err,
                       int model_slot = 0, int frame_slot = 0);

// 精修窗批量评分流内异步版（W3 波2 补完）：与同步版同内核同校验同布局，centers
// H2D/内核/scores D2H 全部发射入指定流后即返回（centers 数组须保持有效至流内
// 拷贝完成）。scores_pinned 为调用方 pinned 缓冲（scores_bytes = cand_count*256*2），
// 事件三点位 e_k0/e_k1/e_d2h 分别记录于内核前/内核后/回读后（内核耗时与回读耗时
// 由调用方 elapsed 测量）。spread0_dev 传设备驻留指针（流水化链路直接给 L0 spread
// 设备地址），空指针语义同同步版（用 ctx 驻留）。双工作面按 frame_slot 隔离。
bool run_window_scores_async(GpuContext* ctx, const uint8_t* spread0_dev, int w0, int h0,
                             int pose_id_offset, const WindowCandidate* cands,
                             const int* cand_pose, int cand_count, uint16_t* scores_pinned,
                             size_t scores_bytes, void* e_k0, void* e_k1, void* e_d2h,
                             int model_slot, void* stream_handle, std::string& err,
                             int frame_slot = 0);

}  // namespace vfshape::gpu
