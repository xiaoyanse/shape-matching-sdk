// 粗层三核 CUDA 内核与链路编排（W2 Phase A，语义平移自尖峰 gpu_chain.cu）：
// L1 链 pyrDown→7 抔高斯→最强通道 Sobel 量化→迟滞→spread→sim_full 全图响应，
// 及 L0 链 grad→spread→sim_full（内核 T 参数化，两级复用）。语义双模：
//  SpikeRef     尖峰参考口径（精确 atan2 + 出界特征跳过 + 全图写）——复刻前
//               对拍归因基线，与尖峰 CPU 参考逐位；
//  VendorExact  vendor 逐位口径（W2 拍板项 2·甲复刻）：fastAtan2 复刻（fast_atan2.h）
//               + 线性内存 flat 越界按 vendor 语义回卷（取模索引；尾格越界读在
//               vendor 为 Mat 越界 UB，此处确定性取模）+ template_positions 外置零
//               （vendor dst=Mat::zeros 只写 span 内）。Phase B 引擎接缝的生产口径。
// 纪律：.cu 字符串字面量一律 ASCII（nvcc 按系统代码页词法分析），中文仅注释。
#include "coarse_kernels.hpp"

#include "fast_atan2.h"
#include "gpu_context.hpp"
#include "similarity_lut.h"

#include <cuda_runtime.h>

#include <chrono>
#include <vector>

namespace vfshape::gpu {
namespace {

__device__ constexpr int kBlurKi[7] = {8, 28, 56, 72, 56, 28, 8};  // /256：高斯定点核
__device__ constexpr int kPyrTap[5] = {1, 4, 6, 4, 1};             // /16：pyrDown 5 抔核
constexpr int kBlockX = 128, kBlockY = 4;
constexpr int kT0 = 4, kT1 = 8;  // 引擎两级采样（model.hpp Detector {4,8}）

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
// BORDER_REFLECT_101 下标（pyrDown 默认边界）
__device__ __forceinline__ int reflect101(int v, int n) {
    if (v < 0) v = -v;
    if (v >= n) v = 2 * (n - 1) - v;
    return v;
}
__device__ __forceinline__ int sim_response(int label, uint8_t mask) {
    return max(SIMILARITY_LUT[32 * label + (mask & 15)],
               SIMILARITY_LUT[32 * label + 16 + (mask >> 4)]);
}
__device__ __forceinline__ int saturate_u8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// —— 内核 1a：7 抔高斯（定点核 /256；中间不舍入、末轮一次舍入——与 OpenCV 逐字节一致）——
__global__ void k_blur_h(const uint8_t* src, int w, int h, uint32_t* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        unsigned sum = 0;
        for (int k = 0; k < 7; ++k) {
            const int sx = clampi(x + k - 3, 0, w - 1);
            sum += unsigned(kBlurKi[k]) * src[(size_t(y) * w + sx) * 3 + c];
        }
        dst[(size_t(y) * w + x) * 3 + c] = sum;
    }
}

__global__ void k_blur_v(const uint32_t* src, int w, int h, uint8_t* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        unsigned sum = 32768;  // 末轮单次舍入：总尺度 /256/256
        for (int k = 0; k < 7; ++k) {
            const int sy = clampi(y + k - 3, 0, h - 1);
            sum += unsigned(kBlurKi[k]) * src[(size_t(sy) * w + x) * 3 + c];
        }
        dst[(size_t(y) * w + x) * 3 + c] = uint8_t(saturate_u8(int(sum >> 16)));
    }
}

// —— 内核 1b：pyrDown（5 抔/16 两维，REFLECT_101，单轮舍入）——
__global__ void k_pyrdown(const uint8_t* src, int w, int h, uint8_t* dst, int dw, int dh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    for (int c = 0; c < 3; ++c) {
        int col[5];
        #pragma unroll
        for (int j = 0; j < 5; ++j) {
            const int sy = reflect101(2 * y + j - 2, h);
            int s = 0;
            #pragma unroll
            for (int i = 0; i < 5; ++i) {
                const int sx = reflect101(2 * x + i - 2, w);
                s += kPyrTap[i] * src[(size_t(sy) * w + sx) * 3 + c];
            }
            col[j] = s;
        }
        int total = 0;
        #pragma unroll
        for (int j = 0; j < 5; ++j) total += kPyrTap[j] * col[j];
        dst[(size_t(y) * dw + x) * 3 + c] = uint8_t(saturate_u8((total + 128) >> 8));
    }
}

// —— 内核 1c：最强通道 Sobel + 16→8 桶量化（边界像素桶=0，vendor 同式）——
// kVendor=true：fastAtan2 复刻 + float 域 round-half-even 量化（vendor 逐位）；
// kVendor=false：精确 atan2 + double 域量化（尖峰参考口径）。
template <bool kVendor>
__global__ void k_grad_quant(const uint8_t* smoothed, int w, int h,
                             uint8_t* bucket, float* mag) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t idx = size_t(y) * w + x;
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        bucket[idx] = 0;
        mag[idx] = 0.f;
        return;
    }
    const uint8_t* top = smoothed + (size_t(y - 1) * w) * 3;
    const uint8_t* mid = smoothed + (size_t(y) * w) * 3;
    const uint8_t* bot = smoothed + (size_t(y + 1) * w) * 3;
    const int l = 3 * (x - 1), r = 3 * (x + 1), c0 = 3 * x;
    int best = -1, best_x = 0, best_y = 0;
    #pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        const int gx = top[r + ch] - top[l + ch] + 2 * (mid[r + ch] - mid[l + ch]) +
                       bot[r + ch] - bot[l + ch];
        const int gy = bot[l + ch] - top[l + ch] + 2 * (bot[c0 + ch] - top[c0 + ch]) +
                       bot[r + ch] - top[r + ch];
        const int norm = gx * gx + gy * gy;
        if (norm > best) {
            best = norm;
            best_x = gx;
            best_y = gy;
        }
    }
    mag[idx] = float(best);
    if (kVendor) {
        bucket[idx] = uint8_t(quantize_orientation(fast_atan2_deg(float(best_y), float(best_x))));
    } else {
        float angle = atan2f(float(best_y), float(best_x)) * 57.29577951308232f;
        if (angle < 0.f) angle += 360.f;
        // 尖峰口径：double 域 round-half-even ×16/360 后 &7
        bucket[idx] = uint8_t(__double2int_rn(double(angle) * (16.0 / 360.0)) & 7);
    }
}

// —— 内核 1d：迟滞多数表决（3×3 直方图含边界零票，首最大，≥5 票成码）——
__global__ void k_hysteresis(const uint8_t* bucket, const float* mag, int w, int h,
                             float weak_sq, uint8_t* angle) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t idx = size_t(y) * w + x;
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1 || mag[idx] <= weak_sq) {
        angle[idx] = 0;
        return;
    }
    int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        const uint8_t* row = bucket + (size_t(y + dy) * w + x - 1);
        histogram[row[0]]++;
        histogram[row[1]]++;
        histogram[row[2]]++;
    }
    int max_votes = 0, index = -1;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (max_votes < histogram[i]) {
            index = i;
            max_votes = histogram[i];
        }
    }
    angle[idx] = max_votes >= 5 ? uint8_t(1 << index) : 0;
}

// —— 内核 2：spread（因果 T 窗 OR，可分离两程，右/下方向）——
__global__ void k_spread_h(const uint8_t* src, uint8_t* dst, int w, int h, int T) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint8_t v = 0;
    const int lim = min(T, w - x);
    for (int k = 0; k < lim; ++k) v |= src[size_t(y) * w + x + k];
    dst[size_t(y) * w + x] = v;
}

__global__ void k_spread_v(const uint8_t* src, uint8_t* dst, int w, int h, int T) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint8_t v = 0;
    const int lim = min(T, h - y);
    for (int k = 0; k < lim; ++k) v |= src[size_t(y + k) * w + x];
    dst[size_t(y) * w + x] = v;
}

// —— 内核 3：sim_full 全图响应累加（grid.y=pose，块内线性遍历解算位置）——
// kVendor=false（尖峰口径）：出界特征/出界读跳过，全 W×H 写。
// kVendor=true（vendor 逐位）：线性内存 flat 语义——
//   · template_positions = span_y*W + span_x + 1 之外恒零（vendor Mat::zeros）；
//   · decimated flat 越界（仅右缘整行区，dx = f.x/T+px ≥ W）回卷进 T×T 网格
//     下一格（vendor Mat 行连续续读的等价取模式）；尾格（grid=T*T-1）越界读
//     在 vendor 为分配越界 UB，此处按 T*T 取模确定性化（CPU 参考同式）。
template <bool kVendor>
__global__ void k_sim_full(const uint8_t* spread, int w, int h,
                           const int4* feats, const int* feat_begin, const int* feat_count,
                           const int2* templ_dims, int T, int W, int H, uint16_t* response) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pose = blockIdx.y;
    if (p >= W * H) return;
    const int px = p % W, py = p / W;
    const int begin = feat_begin[pose], count = feat_count[pose];
    const long long cells = (long long)W * H;
    unsigned sum = 0;
    if (kVendor) {
        const int2 d = templ_dims[pose];
        const int wf = (d.x - 1) / T + 1, hf = (d.y - 1) / T + 1;
        const int template_positions = (H - hf) * W + (W - wf) + 1;
        if (p < template_positions) {
            const int cells32 = W * H;
            for (int i = 0; i < count; ++i) {
                const int4 f = feats[begin + i];
                if (f.x < 0 || f.x >= w || f.y < 0 || f.y >= h) continue;  // vendor 同式跳过
                // 线性内存 flat 语义：lm[grid][lm_index+j]，j=p。dx≥W 时 flat 仍在
                // 本格内续位（decimated 行回卷），flat≥cells 才续入下一格——统一
                // 按 (row,pos) 分解取值，与 vendor Mat 行连续续读逐位等价。
                int pos = (f.y / T + py) * W + (f.x / T + px);
                int row = (f.y % T) * T + (f.x % T);
                if (pos >= cells32) {
                    row += pos / cells32;
                    pos %= cells32;
                }
                row %= T * T;  // 尾格越界读（vendor Mat 越界 UB）确定性取模
                const int pixel_y = (pos / W) * T + row / T;
                const int pixel_x = (pos % W) * T + row % T;
                sum += sim_response(f.z, spread[size_t(pixel_y) * w + pixel_x]);
            }
        }
    } else {
        for (int i = 0; i < count; ++i) {
            const int4 f = feats[begin + i];
            const int fx = px * T + f.x, fy = py * T + f.y;
            if (f.x < w && f.y < h && fx < w && fy < h)
                sum += sim_response(f.z, spread[size_t(fy) * w + fx]);
        }
    }
    response[size_t(pose) * W * H + p] = uint16_t(sum);
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

constexpr int kStageCount = 7;
// 阶段序：pyrdown, grad1, spread1, resp1, grad0, spread0, resp0

}  // namespace

bool run_coarse_chain(GpuContext* ctx, int poses, float weak, CoarseMode mode,
                      bool readback_maps, CoarseResult& out, CoarseTiming* timing,
                      std::string& err, bool need_l0_response, int model_slot,
                      int frame_slot) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    const bool slot1 = model_slot == 1;
    const bool features_ok =
        slot1 ? ctx->fine_features_ready[frame_slot] : ctx->features_ready;
    const bool dims_ok = slot1 ? ctx->fine_dims_ready[frame_slot] : ctx->dims_ready;
    const int resident_poses =
        slot1 ? ctx->fine_resident_poses[frame_slot] : ctx->resident_poses;
    if (!features_ok || !dims_ok) {
        err = "features/template dims not resident";
        return false;
    }
    if (poses <= 0 || poses > resident_poses) {
        err = "pose count out of residency";
        return false;
    }
    LevelBuffers &L0 = ctx->face[frame_slot][0], &L1 = ctx->face[frame_slot][1];
    uint16_t* response_dev = ctx->response[frame_slot];
    const int w = ctx->cur_w[frame_slot] ? ctx->cur_w[frame_slot] : L0.w,
              h = ctx->cur_h[frame_slot] ? ctx->cur_h[frame_slot] : L0.h;
    const int w1 = ctx->cur_w[frame_slot] ? w / 2 : L1.w,
              h1 = ctx->cur_h[frame_slot] ? h / 2 : L1.h;
    if (w % kT0 || h % kT0 || w1 % kT1 || h1 % kT1) {
        err = "level dims must be multiples of T (engine padded frame)";
        return false;
    }
    const int W1 = w1 / kT1, H1 = h1 / kT1, W0 = w / kT0, H0 = h / kT0;
    if (size_t(poses) * W0 * H0 > ctx->response_capacity) {
        err = "response capacity insufficient";
        return false;
    }
    const bool vendor = mode == CoarseMode::VendorExact;
    const float weak_sq = weak * weak;
    const dim3 block(kBlockX, kBlockY);
    const auto grid2d = [](int tw, int th) {
        return dim3((tw + kBlockX - 1) / kBlockX, (th + kBlockY - 1) / kBlockY);
    };
    cudaEvent_t events[2 * kStageCount] = {};
    if (timing)
        for (auto& e : events)
            if (cudaEventCreate(&e) != cudaSuccess) {
                err = "event create failed";
                return false;
            }
    struct EventGuard {
        cudaEvent_t* events;
        bool armed;
        ~EventGuard() {
            if (armed)
                for (int i = 0; i < 2 * kStageCount; ++i)
                    if (events[i]) cudaEventDestroy(events[i]);
        }
    } guard{events, timing != nullptr};
    const auto mark = [&](int s) {
        if (timing) cudaEventRecord(events[2 * s], 0);
    };
    const auto mark_end = [&](int s) {
        if (timing) cudaEventRecord(events[2 * s + 1], 0);
    };
    const auto start_wall = std::chrono::steady_clock::now();

    // —— L1 链：pyrDown → blur → grad → hysteresis → spread(T=8) ——
    mark(0);
    k_pyrdown<<<grid2d(w1, h1), block>>>(L0.bgr, w, h, L1.smooth, w1, h1);
    LAUNCH_CHECK();
    mark_end(0);

    mark(1);
    k_blur_h<<<grid2d(w1, h1), block>>>(L1.smooth, w1, h1, L1.blur_tmp);
    k_blur_v<<<grid2d(w1, h1), block>>>(L1.blur_tmp, w1, h1, L1.bgr);
    if (vendor)
        k_grad_quant<true><<<grid2d(w1, h1), block>>>(L1.bgr, w1, h1, L1.bucket, L1.mag);
    else
        k_grad_quant<false><<<grid2d(w1, h1), block>>>(L1.bgr, w1, h1, L1.bucket, L1.mag);
    k_hysteresis<<<grid2d(w1, h1), block>>>(L1.bucket, L1.mag, w1, h1, weak_sq, L1.angle);
    LAUNCH_CHECK();
    mark_end(1);

    mark(2);
    k_spread_h<<<grid2d(w1, h1), block>>>(L1.angle, L1.spread_h, w1, h1, kT1);
    k_spread_v<<<grid2d(w1, h1), block>>>(L1.spread_h, L1.spread, w1, h1, kT1);
    LAUNCH_CHECK();
    mark_end(2);

    // —— sim_full L1（前半 L1 slot；响应缓冲复用，L1 先回读再跑 L0）——
    // feats 驻留缓冲为 Feature4（与 int4 逐字段同布局，upload 侧按 int4 打包）
    const int4* feats_i4 =
        reinterpret_cast<const int4*>(slot1 ? ctx->fine_feats[frame_slot] : ctx->feats);
    const int* feat_begin = slot1 ? ctx->fine_begin[frame_slot] : ctx->feat_begin;
    const int* feat_count = slot1 ? ctx->fine_count[frame_slot] : ctx->feat_count;
    const int2* templ_dims = slot1 ? ctx->fine_dims[frame_slot] : ctx->templ_dims;
    mark(3);
    if (poses > 0) {
        const dim3 grid((W1 * H1 + kBlockX - 1) / kBlockX, poses);
        if (vendor)
            k_sim_full<true><<<grid, kBlockX>>>(L1.spread, w1, h1, feats_i4, feat_begin,
                                                feat_count, templ_dims, kT1, W1, H1,
                                                response_dev);
        else
            k_sim_full<false><<<grid, kBlockX>>>(L1.spread, w1, h1, feats_i4, feat_begin,
                                                 feat_count, templ_dims, kT1, W1, H1,
                                                 response_dev);
        LAUNCH_CHECK();
    }
    mark_end(3);

    out.resp_w[1] = W1;
    out.resp_h[1] = H1;
    out.response[1].assign(size_t(poses), {});
    {
        std::vector<uint16_t> flat(size_t(poses) * W1 * H1);
        CUDA_CHECK(cudaMemcpy(flat.data(), response_dev, flat.size() * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
        for (int id = 0; id < poses; ++id)
            out.response[1][size_t(id)].assign(
                flat.begin() + size_t(id) * W1 * H1, flat.begin() + size_t(id + 1) * W1 * H1);
        if (readback_maps && need_l0_response) {
            out.maps.w[1] = w1;
            out.maps.h[1] = h1;
            out.maps.angle[1].resize(size_t(w1) * h1);
            out.maps.spread[1].resize(size_t(w1) * h1);
            CUDA_CHECK(cudaMemcpy(out.maps.angle[1].data(), L1.angle, size_t(w1) * h1,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(out.maps.spread[1].data(), L1.spread, size_t(w1) * h1,
                                  cudaMemcpyDeviceToHost));
        }
    }

    // —— L0 链：blur → grad → hysteresis → spread(T=4) → sim_full ——
    mark(4);
    k_blur_h<<<grid2d(w, h), block>>>(L0.bgr, w, h, L0.blur_tmp);
    k_blur_v<<<grid2d(w, h), block>>>(L0.blur_tmp, w, h, L0.smooth);
    if (vendor)
        k_grad_quant<true><<<grid2d(w, h), block>>>(L0.smooth, w, h, L0.bucket, L0.mag);
    else
        k_grad_quant<false><<<grid2d(w, h), block>>>(L0.smooth, w, h, L0.bucket, L0.mag);
    k_hysteresis<<<grid2d(w, h), block>>>(L0.bucket, L0.mag, w, h, weak_sq, L0.angle);
    LAUNCH_CHECK();
    mark_end(4);

    mark(5);
    k_spread_h<<<grid2d(w, h), block>>>(L0.angle, L0.spread_h, w, h, kT0);
    k_spread_v<<<grid2d(w, h), block>>>(L0.spread_h, L0.spread, w, h, kT0);
    LAUNCH_CHECK();
    mark_end(5);

    mark(6);
    // 精益模式：跳过 L0 全图响应内核（宿主提取从 spread[0] 自算窗分）；
    // 事件标记保留（timing 口径统一，elapsed≈0）
    if (need_l0_response) {
        const dim3 grid((W0 * H0 + kBlockX - 1) / kBlockX, poses);
        if (vendor)
            k_sim_full<true><<<grid, kBlockX>>>(
                L0.spread, w, h, feats_i4, feat_begin + ctx->pose_capacity,
                feat_count + ctx->pose_capacity, templ_dims + ctx->pose_capacity,
                kT0, W0, H0, response_dev);
        else
            k_sim_full<false><<<grid, kBlockX>>>(
                L0.spread, w, h, feats_i4, feat_begin + ctx->pose_capacity,
                feat_count + ctx->pose_capacity, templ_dims + ctx->pose_capacity,
                kT0, W0, H0, response_dev);
        LAUNCH_CHECK();
    }
    mark_end(6);

    out.resp_w[0] = W0;
    out.resp_h[0] = H0;
    if (need_l0_response) {
        out.response[0].assign(size_t(poses), {});
        std::vector<uint16_t> flat(size_t(poses) * W0 * H0);
        CUDA_CHECK(cudaMemcpy(flat.data(), response_dev, flat.size() * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
        for (int id = 0; id < poses; ++id)
            out.response[0][size_t(id)].assign(
                flat.begin() + size_t(id) * W0 * H0, flat.begin() + size_t(id + 1) * W0 * H0);
    }
    if (readback_maps) {
        out.maps.w[0] = w;
        out.maps.h[0] = h;
        if (need_l0_response) {
            out.maps.angle[0].resize(size_t(w) * h);
            CUDA_CHECK(cudaMemcpy(out.maps.angle[0].data(), L0.angle, size_t(w) * h,
                                  cudaMemcpyDeviceToHost));
        }
        out.maps.spread[0].resize(size_t(w) * h);
        CUDA_CHECK(cudaMemcpy(out.maps.spread[0].data(), L0.spread, size_t(w) * h,
                              cudaMemcpyDeviceToHost));
    }

    if (timing) {
        float stage_ms[kStageCount] = {};
        for (int s = 0; s < kStageCount; ++s)
            if (cudaEventElapsedTime(&stage_ms[s], events[2 * s], events[2 * s + 1]) !=
                cudaSuccess) {
                err = "event elapsed failed";
                return false;
            }
        timing->pyrdown_ms = stage_ms[0];
        timing->grad1_ms = stage_ms[1];
        timing->spread1_ms = stage_ms[2];
        timing->resp1_ms = stage_ms[3];
        timing->grad0_ms = stage_ms[4];
        timing->spread0_ms = stage_ms[5];
        timing->resp0_ms = stage_ms[6];
        timing->wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      start_wall).count();
    }
    return true;
}

bool run_coarse_chain_async(GpuContext* ctx, int poses, float weak, void* stream_handle,
                            uint16_t* resp1_pinned, size_t resp1_bytes, int& resp_w1,
                            int& resp_h1, const uint8_t** spread0_dev, int& spread_w0,
                            int& spread_h0, void* e_chain_begin, void* e_resp1,
                            void* e_chain_end, int model_slot, int frame_slot,
                            std::string& err) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    if (!stream_handle || !resp1_pinned || !e_resp1 || !e_chain_end || !e_chain_begin) {
        err = "null stream/pinned/event";
        return false;
    }
    const bool slot1 = model_slot == 1;
    const bool features_ok =
        slot1 ? ctx->fine_features_ready[frame_slot] : ctx->features_ready;
    const bool dims_ok = slot1 ? ctx->fine_dims_ready[frame_slot] : ctx->dims_ready;
    const int resident_poses =
        slot1 ? ctx->fine_resident_poses[frame_slot] : ctx->resident_poses;
    if (!features_ok || !dims_ok) {
        err = "features/template dims not resident";
        return false;
    }
    if (poses <= 0 || poses > resident_poses) {
        err = "pose count out of residency";
        return false;
    }
    cudaStream_t stream = (cudaStream_t)stream_handle;
    LevelBuffers &L0 = ctx->face[frame_slot][0], &L1 = ctx->face[frame_slot][1];
    const int4* feats_i4 =
        reinterpret_cast<const int4*>(slot1 ? ctx->fine_feats[frame_slot] : ctx->feats);
    const int* feat_begin =
        slot1 ? ctx->fine_begin[frame_slot] : ctx->feat_begin;
    const int* feat_count =
        slot1 ? ctx->fine_count[frame_slot] : ctx->feat_count;
    const int2* templ_dims = slot1 ? ctx->fine_dims[frame_slot] : ctx->templ_dims;
    uint16_t* response_dev = ctx->response[frame_slot];
    const int w = ctx->cur_w[frame_slot] ? ctx->cur_w[frame_slot] : L0.w,
              h = ctx->cur_h[frame_slot] ? ctx->cur_h[frame_slot] : L0.h;
    const int w1 = ctx->cur_w[frame_slot] ? w / 2 : L1.w,
              h1 = ctx->cur_h[frame_slot] ? h / 2 : L1.h;
    if (w % kT0 || h % kT0 || w1 % kT1 || h1 % kT1) {
        err = "level dims must be multiples of T (engine padded frame)";
        return false;
    }
    const int W1 = w1 / kT1, H1 = h1 / kT1, W0 = w / kT0, H0 = h / kT0;
    if (size_t(poses) * W0 * H0 > ctx->response_capacity) {
        err = "response capacity insufficient";
        return false;
    }
    if (size_t(resp1_bytes) != size_t(poses) * W1 * H1 * sizeof(uint16_t)) {
        err = "resp1 pinned size mismatch";
        return false;
    }
    const float weak_sq = weak * weak;
    const dim3 block(kBlockX, kBlockY);
    const auto grid2d = [](int tw, int th) {
        return dim3((tw + kBlockX - 1) / kBlockX, (th + kBlockY - 1) / kBlockY);
    };
    cudaEventRecord((cudaEvent_t)e_chain_begin, stream);

    // —— L1 链：pyrDown → blur → grad → hysteresis → spread(T=8)（发射序与
    // 同步版逐项一致，仅目标流不同）——
    k_pyrdown<<<grid2d(w1, h1), block, 0, stream>>>(L0.bgr, w, h, L1.smooth, w1, h1);
    LAUNCH_CHECK();
    k_blur_h<<<grid2d(w1, h1), block, 0, stream>>>(L1.smooth, w1, h1, L1.blur_tmp);
    k_blur_v<<<grid2d(w1, h1), block, 0, stream>>>(L1.blur_tmp, w1, h1, L1.bgr);
    k_grad_quant<true><<<grid2d(w1, h1), block, 0, stream>>>(L1.bgr, w1, h1, L1.bucket, L1.mag);
    k_hysteresis<<<grid2d(w1, h1), block, 0, stream>>>(L1.bucket, L1.mag, w1, h1, weak_sq,
                                                       L1.angle);
    LAUNCH_CHECK();
    k_spread_h<<<grid2d(w1, h1), block, 0, stream>>>(L1.angle, L1.spread_h, w1, h1, kT1);
    k_spread_v<<<grid2d(w1, h1), block, 0, stream>>>(L1.spread_h, L1.spread, w1, h1, kT1);
    LAUNCH_CHECK();

    // —— sim_full L1 + resp1 异步回读（L0 链之前入队，宿主扫描与 L0 链重叠）——
    if (poses > 0) {
        const dim3 grid((W1 * H1 + kBlockX - 1) / kBlockX, poses);
        k_sim_full<true><<<grid, kBlockX, 0, stream>>>(L1.spread, w1, h1, feats_i4, feat_begin,
                                                       feat_count, templ_dims, kT1, W1, H1,
                                                       response_dev);
        LAUNCH_CHECK();
    }
    resp_w1 = W1;
    resp_h1 = H1;
    CUDA_CHECK(cudaMemcpyAsync(resp1_pinned, response_dev,
                               size_t(poses) * W1 * H1 * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    cudaEventRecord((cudaEvent_t)e_resp1, stream);

    // —— L0 链：blur → grad → hysteresis → spread(T=4)，lean 无 L0 全图响应 ——
    k_blur_h<<<grid2d(w, h), block, 0, stream>>>(L0.bgr, w, h, L0.blur_tmp);
    k_blur_v<<<grid2d(w, h), block, 0, stream>>>(L0.blur_tmp, w, h, L0.smooth);
    k_grad_quant<true><<<grid2d(w, h), block, 0, stream>>>(L0.smooth, w, h, L0.bucket, L0.mag);
    k_hysteresis<<<grid2d(w, h), block, 0, stream>>>(L0.bucket, L0.mag, w, h, weak_sq, L0.angle);
    LAUNCH_CHECK();
    k_spread_h<<<grid2d(w, h), block, 0, stream>>>(L0.angle, L0.spread_h, w, h, kT0);
    k_spread_v<<<grid2d(w, h), block, 0, stream>>>(L0.spread_h, L0.spread, w, h, kT0);
    LAUNCH_CHECK();

    *spread0_dev = L0.spread;
    spread_w0 = w;
    spread_h0 = h;
    cudaEventRecord((cudaEvent_t)e_chain_end, stream);
    return true;
}

}  // namespace vfshape::gpu
