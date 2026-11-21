// 精修窗批量评分内核（W3 波1）：尖峰 k_sim_windows 平移 + vendor 闭式语义复刻，
// 与宿主 accumulate_window 逐位一致（语义基准 coarse_extract.cpp；行号引
// vendor/shape_based_matching/line2Dup.cpp）。宿主 LM 预烙查表值在设备由
// SIMILARITY_LUT 现算（同 responseLookup 低/高半字节取大），逐位等值。
// 纪律：字符串字面量一律 ASCII（nvcc 按系统代码页词法分析），中文仅注释。
#include "fine_windows.hpp"

#include "gpu_context.hpp"
#include "similarity_lut.h"

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace vfshape::gpu {
namespace {

constexpr int kT0 = 4;  // L0 采样档（引擎 Detector 两级 {4,8}）

__device__ __forceinline__ int sim_response(int label, uint8_t mask) {
    return max(SIMILARITY_LUT[32 * label + (mask & 15)],
               SIMILARITY_LUT[32 * label + 16 + (mask >> 4)]);
}

// blockIdx.x=候选，dim3(16,16)=窗格。闭式索引等价宿主 LM 镜像布局：格 (row,col)
// 对特征 (fx,fy) 读线性相位平面 pos=(fy/T+row)*W+(fx/T)+col，越平面尾 flat 续读
// 下一相位行（镜像语义 = 回卷相位行 0），相位 ≥T*T 取模（Mat 尾 vendor UB 的
// W2-A 先例确定化，k_sim_full 同式）。
__global__ void k_window_scores(const uint8_t* spread, int w0, int h0, const Feature4* feats,
                                const int* feat_begin, const int* feat_count,
                                const int4* centers, uint16_t* scores) {
    const int col = threadIdx.x, row = threadIdx.y;
    const int4 center = centers[blockIdx.x];
    const int begin = feat_begin[center.z], count = feat_count[center.z];
    const int offset_x = (center.x / kT0 - 8) * kT0;
    const int offset_y = (center.y / kT0 - 8) * kT0;
    const int W = w0 / kT0, Hc = h0 / kT0, plane = W * Hc;
    unsigned sum = 0;
    for (int i = 0; i < count; ++i) {
        const Feature4 f = feats[begin + i];
        const int fx = f.x + offset_x, fy = f.y + offset_y;
        if (fx < 0 || fy < 0 || fx >= w0 || fy >= h0) continue;  // vendor :942
        int pos = (fy / kT0 + row) * W + (fx / kT0) + col;
        int phase = (fy % kT0) * kT0 + (fx % kT0);
        if (pos >= plane) {
            phase += pos / plane;
            pos %= plane;
        }
        phase %= kT0 * kT0;
        const int pixel_y = (pos / W) * kT0 + phase / kT0;
        const int pixel_x = (pos % W) * kT0 + phase % kT0;
        sum += sim_response(f.label, spread[size_t(pixel_y) * w0 + pixel_x]);
    }
    scores[blockIdx.x * 256 + row * 16 + col] = uint16_t(sum);
}

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        const cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                               \
            err = std::string("CUDA ") + cudaGetErrorString(status) + " @ " #call; \
            return false;                                                          \
        }                                                                          \
    } while (0)
#define LAUNCH_CHECK()                                                     \
    do {                                                                   \
        const cudaError_t status = cudaGetLastError();                     \
        if (status != cudaSuccess) {                                       \
            err = std::string("CUDA launch ") + cudaGetErrorString(status); \
            return false;                                                  \
        }                                                                  \
    } while (0)

}  // namespace

bool run_window_scores(GpuContext* ctx, const uint8_t* spread0_dev, int w0, int h0,
                       const void* feats_dev, const int* feat_begin_dev,
                       const int* feat_count_dev, int pose_id_offset,
                       const WindowCandidate* cands, const int* cand_pose, int cand_count,
                       std::vector<uint16_t>& host_scores, double* kernel_ms, std::string& err,
                       int model_slot, int frame_slot) {
    host_scores.clear();
    if (kernel_ms) *kernel_ms = 0.0;
    if (!ctx) {
        err = "null context";
        return false;
    }
    const bool slot1 = model_slot == 1;
    if (!(slot1 ? ctx->fine_features_ready[frame_slot] : ctx->features_ready)) {
        err = "features not resident";
        return false;
    }
    if (cand_count < 0) {
        err = "negative candidate count";
        return false;
    }
    if (cand_count > ctx->candidate_capacity) {
        err = "window candidate capacity exceeded";
        return false;
    }
    if (cand_count == 0) return true;
    if (!cands || !cand_pose) {
        err = "null candidate arrays";
        return false;
    }
    if (w0 <= 0 || h0 <= 0 || w0 % kT0 || h0 % kT0) {
        err = "window frame dims must be positive multiples of coarse T";
        return false;
    }
    const int cur_w = ctx->cur_w[frame_slot] ? ctx->cur_w[frame_slot]
                                             : ctx->face[frame_slot][0].w;
    const int cur_h = ctx->cur_h[frame_slot] ? ctx->cur_h[frame_slot]
                                             : ctx->face[frame_slot][0].h;
    if (w0 != cur_w || h0 != cur_h) {
        err = "window frame dims mismatch context L0 level";
        return false;
    }
    std::vector<int4> packed(cand_count);
    for (int i = 0; i < cand_count; ++i) {
        if (cands[i].x < 0 || cands[i].x >= w0 || cands[i].y < 0 || cands[i].y >= h0) {
            err = "window center out of frame";
            return false;
        }
        const int slot = cand_pose[i] + pose_id_offset;
        if (slot < 0 || slot >= ctx->pose_capacity * 2) {
            err = "candidate pose slot out of residency";
            return false;
        }
        packed[size_t(i)] = make_int4(cands[i].x, cands[i].y, slot, 0);
    }
    struct EventGuard {
        cudaEvent_t a = nullptr, b = nullptr;
        ~EventGuard() {
            if (a) cudaEventDestroy(a);
            if (b) cudaEventDestroy(b);
        }
    } guard;
    if (kernel_ms &&
        (cudaEventCreate(&guard.a) != cudaSuccess || cudaEventCreate(&guard.b) != cudaSuccess)) {
        err = "event create failed";
        return false;
    }
    CUDA_CHECK(cudaMemcpy(ctx->centers[frame_slot], packed.data(),
                          sizeof(int4) * size_t(cand_count), cudaMemcpyHostToDevice));
    if (kernel_ms) cudaEventRecord(guard.a, 0);
    k_window_scores<<<cand_count, dim3(16, 16)>>>(
        spread0_dev ? spread0_dev : ctx->face[frame_slot][0].spread, w0, h0,
        feats_dev ? static_cast<const Feature4*>(feats_dev)
                  : (slot1 ? ctx->fine_feats[frame_slot] : ctx->feats),
        feat_begin_dev ? feat_begin_dev
                       : (slot1 ? ctx->fine_begin[frame_slot] : ctx->feat_begin),
        feat_count_dev ? feat_count_dev
                       : (slot1 ? ctx->fine_count[frame_slot] : ctx->feat_count),
        ctx->centers[frame_slot], ctx->window_scores[frame_slot]);
    LAUNCH_CHECK();
    if (kernel_ms) cudaEventRecord(guard.b, 0);
    host_scores.resize(size_t(cand_count) * 256);
    CUDA_CHECK(cudaMemcpy(host_scores.data(), ctx->window_scores[frame_slot],
                          host_scores.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (kernel_ms) {
        float event_ms = 0.f;
        if (cudaEventElapsedTime(&event_ms, guard.a, guard.b) != cudaSuccess) {
            err = "event elapsed failed";
            return false;
        }
        *kernel_ms = double(event_ms);
    }
    return true;
}

bool run_window_scores_async(GpuContext* ctx, const uint8_t* spread0_dev, int w0, int h0,
                             int pose_id_offset, const WindowCandidate* cands,
                             const int* cand_pose, int cand_count, uint16_t* scores_pinned,
                             size_t scores_bytes, void* e_k0, void* e_k1, void* e_d2h,
                             int model_slot, void* stream_handle, std::string& err,
                             int frame_slot) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    const bool slot1 = model_slot == 1;
    if (!(slot1 ? ctx->fine_features_ready[frame_slot] : ctx->features_ready)) {
        err = "features not resident";
        return false;
    }
    if (cand_count < 0) {
        err = "negative candidate count";
        return false;
    }
    if (cand_count > ctx->candidate_capacity) {
        err = "window candidate capacity exceeded";
        return false;
    }
    if (cand_count == 0) return true;
    if (!cands || !cand_pose || !scores_pinned || !stream_handle) {
        err = "null candidate/pinned/stream";
        return false;
    }
    if (scores_bytes != size_t(cand_count) * 256 * sizeof(uint16_t)) {
        err = "scores pinned size mismatch";
        return false;
    }
    if (w0 <= 0 || h0 <= 0 || w0 % kT0 || h0 % kT0) {
        err = "window frame dims must be positive multiples of coarse T";
        return false;
    }
    const int cur_w = ctx->cur_w[frame_slot] ? ctx->cur_w[frame_slot]
                                             : ctx->face[frame_slot][0].w;
    const int cur_h = ctx->cur_h[frame_slot] ? ctx->cur_h[frame_slot]
                                             : ctx->face[frame_slot][0].h;
    if (w0 != cur_w || h0 != cur_h) {
        err = "window frame dims mismatch context L0 level";
        return false;
    }
    std::vector<int4> packed(cand_count);
    for (int i = 0; i < cand_count; ++i) {
        if (cands[i].x < 0 || cands[i].x >= w0 || cands[i].y < 0 || cands[i].y >= h0) {
            err = "window center out of frame";
            return false;
        }
        const int slot = cand_pose[i] + pose_id_offset;
        if (slot < 0 || slot >= ctx->pose_capacity * 2) {
            err = "candidate pose slot out of residency";
            return false;
        }
        packed[size_t(i)] = make_int4(cands[i].x, cands[i].y, slot, 0);
    }
    cudaStream_t stream = (cudaStream_t)stream_handle;
    const Feature4* feats = slot1 ? ctx->fine_feats[frame_slot] : ctx->feats;
    const int* feat_begin = slot1 ? ctx->fine_begin[frame_slot] : ctx->feat_begin;
    const int* feat_count = slot1 ? ctx->fine_count[frame_slot] : ctx->feat_count;
    CUDA_CHECK(cudaMemcpyAsync(ctx->centers[frame_slot], packed.data(),
                               sizeof(int4) * size_t(cand_count), cudaMemcpyHostToDevice,
                               stream));
    cudaEventRecord((cudaEvent_t)e_k0, stream);
    k_window_scores<<<cand_count, dim3(16, 16), 0, stream>>>(
        spread0_dev ? spread0_dev : ctx->face[frame_slot][0].spread, w0, h0, feats, feat_begin,
        feat_count, ctx->centers[frame_slot], ctx->window_scores[frame_slot]);
    LAUNCH_CHECK();
    cudaEventRecord((cudaEvent_t)e_k1, stream);
    CUDA_CHECK(cudaMemcpyAsync(scores_pinned, ctx->window_scores[frame_slot], scores_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cudaEventRecord((cudaEvent_t)e_d2h, stream);
    return true;
}

}  // namespace vfshape::gpu
