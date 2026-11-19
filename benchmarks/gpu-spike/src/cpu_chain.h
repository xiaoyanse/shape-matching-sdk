#ifndef GPU_SPIKE_CPU_CHAIN_H
#define GPU_SPIKE_CPU_CHAIN_H
// CPU 单线程参考链：vendor line2Dup 语义直译（GaussianBlur/phase/pyrDown/resize
// 用 OpenCV 同参调用，特征通道择优/迟滞/spread/累加为标量直译）。
#include "common.h"

namespace gpuspike {

void cpu_gradient_quantize(const uint8_t* bgr, int w, int h, float weak, Plane& angle);
void cpu_spread(const Plane& angle, int T, Plane& spread);
void cpu_similarity_full(const Plane& spread, const TemplLevel& level, std::vector<uint16_t>& dst);
// 批量精修窗分数（每候选 256 值），供共享 refine_from_scores
void cpu_compute_windows(const Plane& spread0, const std::vector<PoseModel>& poses,
                         const std::vector<CoarseCand>& candidates, int w0, int h0,
                         std::vector<uint16_t>& scores);

void run_chain_cpu(const uint8_t* padded_bgr, int w, int h, float weak,
                   const std::vector<PoseModel>& poses,
                   float coarse_th, float fine_th, ChainOutput& out);

}  // namespace gpuspike

#endif
