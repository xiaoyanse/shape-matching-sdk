#include "gpu_chain.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <vector>

namespace gpuspike {
// SIMILARITY_LUT 经 common.h 以 __constant__ 引入（__CUDACC__ 分支），全设备可见

namespace {

__device__ constexpr int kBlurKi[7] = {8, 28, 56, 72, 56, 28, 8};  // /256：高斯定点核
__device__ constexpr int kPyrTap[5] = {1, 4, 6, 4, 1};            // /16：pyrDown 5 抔核
constexpr int kBlockX = 128, kBlockY = 4;

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
    float angle = atan2f(float(best_y), float(best_x)) * 57.29577951308232f;
    if (angle < 0.f) angle += 360.f;
    // vendor convertTo(CV_8U, 16/360)：round-half-even + &7（double 域防精度漂移）
    const int q = int(rint(double(angle) * (16.0 / 360.0))) & 7;
    bucket[idx] = uint8_t(q);
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

// —— 内核 3a：全图响应累加（L1，T=8；grid.y=pose，块内线性遍历解算位置）——
__global__ void k_sim_full(const uint8_t* spread, int w, int h,
                           const int4* feats, const int* feat_begin, const int* feat_count,
                           int W, int H, uint16_t* response) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pose = blockIdx.y;
    if (p >= W * H) return;
    const int px = p % W, py = p / W;
    const int begin = feat_begin[pose], count = feat_count[pose];
    constexpr int T = 8;
    unsigned sum = 0;
    for (int i = 0; i < count; ++i) {
        const int4 f = feats[begin + i];
        const int fx = px * T + f.x, fy = py * T + f.y;
        if (f.x < w && f.y < h && fx < w && fy < h)
            sum += sim_response(f.z, spread[size_t(fy) * w + fx]);
    }
    response[size_t(pose) * W * H + p] = uint16_t(sum);
}

// —— 内核 3b：16×16 精修窗批量（L0，T=4；blockIdx.x=候选，256 线程=窗格）——
// 中心 (cx,cy) 已按 refine 夹取规则在宿主侧算好，设备只算窗格分数
__global__ void k_sim_windows(const uint8_t* spread, int w, int h,
                              const int4* feats, const int* feat_begin, const int* feat_count,
                              const int4* centers, uint16_t* scores) {
    const int col = threadIdx.x, row = threadIdx.y;
    const int4 center = centers[blockIdx.x];
    const int pose = center.z, cx = center.x, cy = center.y;
    constexpr int T = 4;
    const int offset_x = (cx / T - 8) * T;
    const int offset_y = (cy / T - 8) * T;
    const int begin = feat_begin[pose], count = feat_count[pose];
    unsigned sum = 0;
    for (int i = 0; i < count; ++i) {
        const int4 f = feats[begin + i];
        const int bx = f.x + offset_x, by = f.y + offset_y;
        if (bx < 0 || by < 0 || bx >= w || by >= h) continue;
        const int sx = bx + col * T, sy = by + row * T;
        if (sx >= w || sy >= h) continue;
        sum += sim_response(f.z, spread[size_t(sy) * w + sx]);
    }
    scores[blockIdx.x * 256 + row * 16 + col] = uint16_t(sum);
}

struct LevelBuffers {
    uint8_t *bgr = nullptr, *smooth = nullptr;
    uint32_t* blur_tmp = nullptr;
    uint8_t *bucket = nullptr, *angle = nullptr, *spread = nullptr, *spread_h = nullptr;
    float* mag = nullptr;
    int w = 0, h = 0;
};

}  // namespace

struct GpuChainContext {
    LevelBuffers level[2];  // [0]=L0 全分辨率档，[1]=L1 半分辨率档
    int4* feats = nullptr;
    int *feat_begin = nullptr, *feat_count = nullptr;  // 前半 L1、后半 L0
    uint16_t* response = nullptr;
    int4* centers = nullptr;
    uint16_t* window_scores = nullptr;
    size_t response_capacity = 0;
    int pose_capacity = 0;
    bool features_ready = false;
    cudaEvent_t events[static_cast<size_t>(Stage::Count) * 2] = {};
};

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        const cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                               \
            err = std::string("CUDA ") + cudaGetErrorString(status) + " @ " #call; \
            return false;                                                          \
        }                                                                          \
    } while (0)
#define LAUNCH_CHECK()                                                    \
    do {                                                                  \
        const cudaError_t status = cudaGetLastError();                    \
        if (status != cudaSuccess) {                                      \
            err = std::string("CUDA launch ") + cudaGetErrorString(status); \
            return false;                                                 \
        }                                                                 \
    } while (0)

GpuChainContext* gpu_create_context(int max_w, int max_h, int max_poses, int max_feats,
                                    int max_candidates, int max_positions, std::string& err) {
    auto* ctx = new (std::nothrow) GpuChainContext();
    if (!ctx) {
        err = "context alloc failed";
        return nullptr;
    }
    ctx->pose_capacity = max_poses;
    for (int level = 0; level < 2; ++level) {
        auto& b = ctx->level[level];
        b.w = level ? max_w / 2 : max_w;
        b.h = level ? max_h / 2 : max_h;
        const size_t pixels = size_t(b.w) * b.h;
        if (cudaMalloc(&b.bgr, pixels * 3) != cudaSuccess ||
            cudaMalloc(&b.smooth, pixels * 3) != cudaSuccess ||
            cudaMalloc(&b.blur_tmp, pixels * 3 * 4) != cudaSuccess ||
            cudaMalloc(&b.bucket, pixels) != cudaSuccess ||
            cudaMalloc(&b.angle, pixels) != cudaSuccess ||
            cudaMalloc(&b.spread, pixels) != cudaSuccess ||
            cudaMalloc(&b.spread_h, pixels) != cudaSuccess ||
            cudaMalloc(&b.mag, pixels * 4) != cudaSuccess) {
            gpu_destroy_context(ctx);
            err = "device alloc failed";
            return nullptr;
        }
    }
    ctx->response_capacity = size_t(max_poses) * max_positions;
    if (cudaMalloc(&ctx->feats, sizeof(int4) * max_feats) != cudaSuccess ||
        cudaMalloc(&ctx->feat_begin, sizeof(int) * max_poses * 2) != cudaSuccess ||
        cudaMalloc(&ctx->feat_count, sizeof(int) * max_poses * 2) != cudaSuccess ||
        cudaMalloc(&ctx->response, sizeof(uint16_t) * ctx->response_capacity) != cudaSuccess ||
        cudaMalloc(&ctx->centers, sizeof(int4) * max_candidates) != cudaSuccess ||
        cudaMalloc(&ctx->window_scores, sizeof(uint16_t) * 256 * max_candidates) != cudaSuccess) {
        gpu_destroy_context(ctx);
        err = "device alloc failed (feats/response)";
        return nullptr;
    }
    for (auto& event : ctx->events) {
        if (cudaEventCreate(&event) != cudaSuccess) {
            gpu_destroy_context(ctx);
            err = "event create failed";
            return nullptr;
        }
    }
    return ctx;
}

void gpu_destroy_context(GpuChainContext* ctx) {
    if (!ctx) return;
    for (auto& b : ctx->level) {
        cudaFree(b.bgr);
        cudaFree(b.smooth);
        cudaFree(b.blur_tmp);
        cudaFree(b.bucket);
        cudaFree(b.angle);
        cudaFree(b.spread);
        cudaFree(b.spread_h);
        cudaFree(b.mag);
    }
    cudaFree(ctx->feats);
    cudaFree(ctx->feat_begin);
    cudaFree(ctx->feat_count);
    cudaFree(ctx->response);
    cudaFree(ctx->centers);
    cudaFree(ctx->window_scores);
    for (auto& event : ctx->events)
        if (event) cudaEventDestroy(event);
    delete ctx;
}

bool gpu_upload_features(GpuChainContext* ctx, const std::vector<PoseModel>& poses,
                         std::string& err) {
    const size_t n = poses.size();
    if (int(n) > ctx->pose_capacity) {
        err = "pose count exceeds capacity";
        return false;
    }
    std::vector<int4> host_feats;
    std::vector<int> begin(int(n) * 2), count(int(n) * 2);
    for (int level = 1; level >= 0; --level) {  // 先 L1 后 L0 拼接
        for (size_t id = 0; id < n; ++id) {
            const int slot = level == 1 ? int(id) : ctx->pose_capacity + int(id);
            begin[slot] = int(host_feats.size());
            const auto& feats = poses[id].level[level].feats;
            count[slot] = int(feats.size());
            for (const auto& f : feats) host_feats.push_back({f.x, f.y, f.label, 0});
        }
    }
    CUDA_CHECK(cudaMemcpy(ctx->feats, host_feats.data(), sizeof(int4) * host_feats.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->feat_begin, begin.data(), sizeof(int) * begin.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->feat_count, count.data(), sizeof(int) * count.size(),
                          cudaMemcpyHostToDevice));
    ctx->features_ready = true;
    return true;
}

bool gpu_run_chain(GpuChainContext* ctx, const uint8_t* padded_bgr, int w, int h, float weak,
                   const std::vector<PoseModel>& poses, float coarse_th, float fine_th,
                   bool upload, bool correctness, ChainOutput& out, std::string& err) {
    if (!ctx->features_ready) {
        err = "features not resident (call gpu_upload_features first)";
        return false;
    }
    if (w != ctx->level[0].w || h != ctx->level[0].h) {
        err = "chain size mismatch with context";
        return false;
    }
    if (int(poses.size()) > ctx->pose_capacity) {
        err = "pose count exceeds capacity";
        return false;
    }
    const auto start_wall = std::chrono::steady_clock::now();
    const int w1 = ctx->level[1].w, h1 = ctx->level[1].h;
    const float weak_sq = weak * weak;
    LevelBuffers &L0 = ctx->level[0], &L1 = ctx->level[1];
    const dim3 block(kBlockX, kBlockY);
    const auto grid2d = [](int w, int h) {
        return dim3((w + kBlockX - 1) / kBlockX, (h + kBlockY - 1) / kBlockY);
    };
    const auto mark = [ctx](Stage s) { cudaEventRecord(ctx->events[2 * size_t(s)], 0); };
    const auto mark_end = [ctx](Stage s) { cudaEventRecord(ctx->events[2 * size_t(s) + 1], 0); };

    if (upload)
        CUDA_CHECK(cudaMemcpy(L0.bgr, padded_bgr, size_t(w) * h * 3, cudaMemcpyHostToDevice));

    mark(Stage::PyrDown);
    k_pyrdown<<<grid2d(w1, h1), block>>>(L0.bgr, w, h, L1.smooth, w1, h1);
    LAUNCH_CHECK();
    mark_end(Stage::PyrDown);

    mark(Stage::Grad1);
    k_blur_h<<<grid2d(w1, h1), block>>>(L1.smooth, w1, h1, L1.blur_tmp);
    k_blur_v<<<grid2d(w1, h1), block>>>(L1.blur_tmp, w1, h1, L1.bgr);
    k_grad_quant<<<grid2d(w1, h1), block>>>(L1.bgr, w1, h1, L1.bucket, L1.mag);
    k_hysteresis<<<grid2d(w1, h1), block>>>(L1.bucket, L1.mag, w1, h1, weak_sq, L1.angle);
    LAUNCH_CHECK();
    mark_end(Stage::Grad1);

    mark(Stage::Spread1);
    k_spread_h<<<grid2d(w1, h1), block>>>(L1.angle, L1.spread_h, w1, h1, kT[1]);
    k_spread_v<<<grid2d(w1, h1), block>>>(L1.spread_h, L1.spread, w1, h1, kT[1]);
    LAUNCH_CHECK();
    mark_end(Stage::Spread1);

    out.resp_W = w1 / kT[1];
    out.resp_H = h1 / kT[1];
    if (size_t(poses.size()) * out.resp_W * out.resp_H > ctx->response_capacity) {
        err = "response capacity insufficient";
        return false;
    }
    // 响应累加：feat_begin/count 前半段=L1（gpu_upload_features 布局）
    mark(Stage::Resp1);
    if (!poses.empty()) {
        k_sim_full<<<dim3((out.resp_W * out.resp_H + kBlockX - 1) / kBlockX, int(poses.size())),
                    kBlockX>>>(L1.spread, w1, h1, ctx->feats, ctx->feat_begin, ctx->feat_count,
                               out.resp_W, out.resp_H, ctx->response);
        LAUNCH_CHECK();
    }
    mark_end(Stage::Resp1);

    // 响应图回读 + 共享候选提取（宿主侧）
    const auto extract_start = std::chrono::steady_clock::now();
    out.response.assign(poses.size(), {});
    std::vector<uint16_t> flat(size_t(poses.size()) * out.resp_W * out.resp_H);
    CUDA_CHECK(cudaMemcpy(flat.data(), ctx->response, flat.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    for (size_t id = 0; id < poses.size(); ++id) {
        const auto& level1 = poses[id].level[1];
        if (!template_fits(poses[id], w, h, w1, h1) || level1.feats.empty()) continue;
        out.response[id].assign(flat.begin() + size_t(id) * out.resp_W * out.resp_H,
                                flat.begin() + size_t(id + 1) * out.resp_W * out.resp_H);
    }
    out.candidates = extract_candidates(out.response, out.resp_W, out.resp_H, poses, coarse_th);
    const auto extract_end = std::chrono::steady_clock::now();
    out.host_extract_ms +=
        std::chrono::duration<double, std::milli>(extract_end - extract_start).count();

    mark(Stage::Grad0);
    k_blur_h<<<grid2d(w, h), block>>>(L0.bgr, w, h, L0.blur_tmp);
    k_blur_v<<<grid2d(w, h), block>>>(L0.blur_tmp, w, h, L0.smooth);
    k_grad_quant<<<grid2d(w, h), block>>>(L0.smooth, w, h, L0.bucket, L0.mag);
    k_hysteresis<<<grid2d(w, h), block>>>(L0.bucket, L0.mag, w, h, weak_sq, L0.angle);
    LAUNCH_CHECK();
    mark_end(Stage::Grad0);

    mark(Stage::Spread0);
    k_spread_h<<<grid2d(w, h), block>>>(L0.angle, L0.spread_h, w, h, kT[0]);
    k_spread_v<<<grid2d(w, h), block>>>(L0.spread_h, L0.spread, w, h, kT[0]);
    LAUNCH_CHECK();
    mark_end(Stage::Spread0);

    // 精修窗批量：一次上传全部中心 + 单内核 + 单次回读，再走共享精修
    mark(Stage::Resp0);
    if (!out.candidates.empty()) {
        if (out.candidates.size() > size_t(16384)) {
            err = "window candidate budget exceeded";
            return false;
        }
        std::vector<int4> centers(out.candidates.size());
        for (size_t ci = 0; ci < out.candidates.size(); ++ci) {
            const auto& cand = out.candidates[ci];
            const auto& templ = poses[size_t(cand.pose_id)].level[0];
            int x = 0, y = 0;
            clamp_center(cand, templ, w, h, x, y);
            centers[ci] = {x, y, cand.pose_id, 0};
        }
        CUDA_CHECK(cudaMemcpy(ctx->centers, centers.data(), sizeof(int4) * centers.size(),
                              cudaMemcpyHostToDevice));
        k_sim_windows<<<int(centers.size()), dim3(16, 16)>>>(
            L0.spread, w, h, ctx->feats, ctx->feat_begin + ctx->pose_capacity,
            ctx->feat_count + ctx->pose_capacity, ctx->centers, ctx->window_scores);
        LAUNCH_CHECK();
        std::vector<uint16_t> scores(centers.size() * 256);
        CUDA_CHECK(cudaMemcpy(scores.data(), ctx->window_scores,
                              scores.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        out.refined = refine_from_scores(out.candidates, poses, w, h, fine_th, scores);
    }
    mark_end(Stage::Resp0);

    // 阶段耗时（设备事件）+ 墙钟
    for (size_t s = 0; s < static_cast<size_t>(Stage::Count); ++s) {
        float ms = 0;
        cudaEventElapsedTime(&ms, ctx->events[2 * s], ctx->events[2 * s + 1]);
        out.times.ms[s] = ms;
    }
    out.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                            start_wall)
                      .count();
    if (correctness) {
        out.maps[1].angle.w = w1;
        out.maps[1].angle.h = h1;
        out.maps[1].angle.data.resize(size_t(w1) * h1);
        CUDA_CHECK(cudaMemcpy(out.maps[1].angle.data.data(), L1.angle, size_t(w1) * h1,
                              cudaMemcpyDeviceToHost));
        out.maps[1].spread.data.resize(size_t(w1) * h1);
        CUDA_CHECK(cudaMemcpy(out.maps[1].spread.data.data(), L1.spread, size_t(w1) * h1,
                              cudaMemcpyDeviceToHost));
        out.maps[0].angle.w = w;
        out.maps[0].angle.h = h;
        out.maps[0].angle.data.resize(size_t(w) * h);
        CUDA_CHECK(cudaMemcpy(out.maps[0].angle.data.data(), L0.angle, size_t(w) * h,
                              cudaMemcpyDeviceToHost));
        out.maps[0].spread.data.resize(size_t(w) * h);
        CUDA_CHECK(cudaMemcpy(out.maps[0].spread.data.data(), L0.spread, size_t(w) * h,
                              cudaMemcpyDeviceToHost));
    }
    return true;
}

bool gpu_device_info(std::string& name, int& major, int& minor, std::string& err) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    name = prop.name;
    major = prop.major;
    minor = prop.minor;
    return true;
}

}  // namespace gpuspike
