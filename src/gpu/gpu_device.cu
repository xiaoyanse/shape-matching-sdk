// GPU 设备面实现（W1 骨架 + W2 模板尺寸驻留）：CUDA 上下文/POD 驻留/帧上传/LUT
// 驻留；匹配内核在 coarse_kernels.cu。本文件不 include 引擎头（model.hpp/OpenCV/
// vendor），nvcc 编译坑面最小。纪律（尖峰 README 环境坑表）：nvcc 前端按系统代码
// 页词法分析 .cu 字符串字面量——本文件字面量一律 ASCII，中文仅出现在注释。
#include "gpu_device.hpp"
#include "gpu_context.hpp"
#include "similarity_lut.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <new>
#include <string>
#include <vector>

namespace vfshape::gpu {
namespace {

// SIMILARITY_LUT 经 similarity_lut.h 以 __constant__ 引入（__CUDACC__ 分支），
// 全设备可见；模块加载即带初始映像，lut_upload 仍显式上传以钉住同源口径

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        const cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                               \
            err = std::string("CUDA ") + cudaGetErrorString(status) + " @ " #call; \
            return false;                                                          \
        }                                                                          \
    } while (0)

Feature4* slot_feats(GpuContext* ctx, int slot, int fs) {
    return slot ? ctx->fine_feats[fs] : ctx->feats;
}
int* slot_begin(GpuContext* ctx, int slot, int fs) {
    return slot ? ctx->fine_begin[fs] : ctx->feat_begin;
}
int* slot_count(GpuContext* ctx, int slot, int fs) {
    return slot ? ctx->fine_count[fs] : ctx->feat_count;
}
int2* slot_dims(GpuContext* ctx, int slot, int fs) {
    return slot ? ctx->fine_dims[fs] : ctx->templ_dims;
}

bool slot_ready(const GpuContext* ctx, int slot, int fs) {
    return slot ? (ctx->fine_features_ready[fs] && ctx->fine_dims_ready[fs])
                : (ctx->features_ready && ctx->dims_ready);
}
int slot_resident_poses(const GpuContext* ctx, int slot, int fs) {
    return slot ? ctx->fine_resident_poses[fs] : ctx->resident_poses;
}

bool validate_features_args(const GpuContext* ctx, int feat_total, int layout_len, int poses,
                            std::string& err) {
    if (poses <= 0 || poses > ctx->pose_capacity) {
        err = "pose count out of capacity";
        return false;
    }
    if (feat_total <= 0 || feat_total > ctx->feat_capacity) {
        err = "feature count out of capacity";
        return false;
    }
    if (layout_len != ctx->pose_capacity * 2) {
        err = "begin/count layout length mismatch";
        return false;
    }
    return true;
}

bool validate_slot(int model_slot, std::string& err) {
    if (model_slot != 0 && model_slot != 1) {
        err = "model slot out of range";
        return false;
    }
    return true;
}

// 特征驻留共享实现（同步/异步同校验同打包；stream=nullptr 走同步拷贝，W2 行为
// 不变；fine 槽按 frame_slot 双份，流水化双 job 并存互不覆盖）
bool upload_features_impl(GpuContext* ctx, const FeatureRecord* feats, int feat_total,
                          const int* begin, const int* count, int layout_len, int poses,
                          int model_slot, int frame_slot, cudaStream_t stream,
                          std::string& err) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    if (!validate_slot(model_slot, err) ||
        !validate_features_args(ctx, feat_total, layout_len, poses, err))
        return false;
    if (model_slot && (frame_slot != 0 && frame_slot != 1)) {
        err = "frame slot out of range";
        return false;
    }
    // 宿主 FeatureRecord(12B) 与设备 Feature4(16B, int4 对齐) 布局不同：
    // 打包后上传，readback_features 侧做反向转换——两端口径互为镜像
    std::vector<Feature4> packed;
    packed.reserve(size_t(feat_total));
    for (int i = 0; i < feat_total; ++i)
        packed.push_back({feats[i].x, feats[i].y, feats[i].label, 0});
    const cudaError_t s1 = stream
        ? cudaMemcpyAsync(slot_feats(ctx, model_slot, frame_slot), packed.data(),
                          sizeof(Feature4) * feat_total, cudaMemcpyHostToDevice, stream)
        : cudaMemcpy(slot_feats(ctx, model_slot, frame_slot), packed.data(),
                     sizeof(Feature4) * feat_total, cudaMemcpyHostToDevice);
    const cudaError_t s2 = stream
        ? cudaMemcpyAsync(slot_begin(ctx, model_slot, frame_slot), begin,
                          sizeof(int) * layout_len, cudaMemcpyHostToDevice, stream)
        : cudaMemcpy(slot_begin(ctx, model_slot, frame_slot), begin,
                     sizeof(int) * layout_len, cudaMemcpyHostToDevice);
    const cudaError_t s3 = stream
        ? cudaMemcpyAsync(slot_count(ctx, model_slot, frame_slot), count,
                          sizeof(int) * layout_len, cudaMemcpyHostToDevice, stream)
        : cudaMemcpy(slot_count(ctx, model_slot, frame_slot), count,
                     sizeof(int) * layout_len, cudaMemcpyHostToDevice);
    if (s1 != cudaSuccess || s2 != cudaSuccess || s3 != cudaSuccess) {
        err = std::string("CUDA ") + cudaGetErrorString(s1 != cudaSuccess ? s1
                                            : s2 != cudaSuccess ? s2 : s3);
        return false;
    }
    if (model_slot) {
        ctx->fine_features_ready[frame_slot] = true;
        ctx->fine_resident_poses[frame_slot] = poses;
        ctx->fine_resident_feats[frame_slot] = feat_total;
    } else {
        ctx->features_ready = true;
        ctx->resident_poses = poses;
        ctx->resident_feats = feat_total;
    }
    return true;
}

bool upload_template_dims_impl(GpuContext* ctx, const int* widths, const int* heights, int len,
                               int model_slot, int frame_slot, cudaStream_t stream,
                               std::string& err) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    if (!validate_slot(model_slot, err)) return false;
    if (!widths || !heights) {
        err = "null dims arrays";
        return false;
    }
    if (len != ctx->pose_capacity * 2) {
        err = "template dims length mismatch";
        return false;
    }
    // 每位姿每层模板 (width,height)，slot 布局与 feat_begin/feat_count 同式
    //（前半 L1 slot=id、后半 L0 slot=pose_capacity+id）——vendor similarity 的
    // span 边界与线性内存回卷语义依赖模板宽高（wf/hf），W2 粗层内核驻留此表
    std::vector<int2> packed;
    packed.resize(size_t(len));
    for (int i = 0; i < len; ++i) {
        if (widths[i] <= 0 || heights[i] <= 0) {
            err = "template dims must be positive";
            return false;
        }
        packed[size_t(i)] = make_int2(widths[i], heights[i]);
    }
    const cudaError_t s = stream
        ? cudaMemcpyAsync(slot_dims(ctx, model_slot, frame_slot), packed.data(),
                          sizeof(int2) * size_t(len), cudaMemcpyHostToDevice, stream)
        : cudaMemcpy(slot_dims(ctx, model_slot, frame_slot), packed.data(),
                     sizeof(int2) * size_t(len), cudaMemcpyHostToDevice);
    if (s != cudaSuccess) {
        err = std::string("CUDA ") + cudaGetErrorString(s);
        return false;
    }
    if (model_slot)
        ctx->fine_dims_ready[frame_slot] = true;
    else
        ctx->dims_ready = true;
    return true;
}

}  // namespace

GpuContext* create_context(int max_w, int max_h, int max_poses, int max_feats,
                           int max_candidates, int max_positions, std::string& err) {
    if (max_w <= 0 || max_h <= 0 || max_poses <= 0 || max_feats <= 0 ||
        max_candidates <= 0 || max_positions <= 0) {
        err = "context capacities must be positive";
        return nullptr;
    }
    auto* ctx = new (std::nothrow) GpuContext();
    if (!ctx) {
        err = "context alloc failed";
        return nullptr;
    }
    ctx->pose_capacity = max_poses;
    ctx->feat_capacity = max_feats;
    ctx->candidate_capacity = max_candidates;
    for (int fs = 0; fs < 2; ++fs)
        for (int level = 0; level < 2; ++level) {
            auto& b = ctx->face[fs][level];
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
                destroy_context(ctx);
                err = "device alloc failed";
                return nullptr;
            }
        }
    ctx->response_capacity = size_t(max_poses) * max_positions;
    if (cudaMalloc(&ctx->feats, sizeof(Feature4) * max_feats) != cudaSuccess ||
        cudaMalloc(&ctx->feat_begin, sizeof(int) * max_poses * 2) != cudaSuccess ||
        cudaMalloc(&ctx->feat_count, sizeof(int) * max_poses * 2) != cudaSuccess ||
        cudaMalloc(&ctx->templ_dims, sizeof(int2) * max_poses * 2) != cudaSuccess ||
        cudaMalloc(&ctx->response[0], sizeof(uint16_t) * ctx->response_capacity) !=
            cudaSuccess ||
        cudaMalloc(&ctx->response[1], sizeof(uint16_t) * ctx->response_capacity) !=
            cudaSuccess ||
        cudaMalloc(&ctx->centers[0], sizeof(int4) * max_candidates) != cudaSuccess ||
        cudaMalloc(&ctx->centers[1], sizeof(int4) * max_candidates) != cudaSuccess ||
        cudaMalloc(&ctx->window_scores[0], sizeof(uint16_t) * 256 * max_candidates) !=
            cudaSuccess ||
        cudaMalloc(&ctx->window_scores[1], sizeof(uint16_t) * 256 * max_candidates) !=
            cudaSuccess) {
        destroy_context(ctx);
        err = "device alloc failed (feats/response)";
        return nullptr;
    }
    for (int fs = 0; fs < 2; ++fs) {
        if (cudaMalloc(&ctx->fine_feats[fs], sizeof(Feature4) * max_feats) != cudaSuccess ||
            cudaMalloc(&ctx->fine_begin[fs], sizeof(int) * max_poses * 2) != cudaSuccess ||
            cudaMalloc(&ctx->fine_count[fs], sizeof(int) * max_poses * 2) != cudaSuccess ||
            cudaMalloc(&ctx->fine_dims[fs], sizeof(int2) * max_poses * 2) != cudaSuccess) {
            destroy_context(ctx);
            err = "device alloc failed (fine residency)";
            return nullptr;
        }
    }
    return ctx;
}

void destroy_context(GpuContext* ctx) {
    if (!ctx) return;
    for (int fs = 0; fs < 2; ++fs)
        for (int level = 0; level < 2; ++level) {
            auto& b = ctx->face[fs][level];
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
    cudaFree(ctx->templ_dims);
    for (int fs = 0; fs < 2; ++fs) {
        cudaFree(ctx->response[fs]);
        cudaFree(ctx->centers[fs]);
        cudaFree(ctx->window_scores[fs]);
        cudaFree(ctx->fine_feats[fs]);
        cudaFree(ctx->fine_begin[fs]);
        cudaFree(ctx->fine_count[fs]);
        cudaFree(ctx->fine_dims[fs]);
    }
    delete ctx;
}

bool upload_features(GpuContext* ctx, const FeatureRecord* feats, int feat_total,
                     const int* begin, const int* count, int layout_len, int poses,
                     std::string& err, int model_slot, int frame_slot) {
    return upload_features_impl(ctx, feats, feat_total, begin, count, layout_len, poses,
                                model_slot, frame_slot, nullptr, err);
}

bool readback_features(const GpuContext* ctx, std::vector<FeatureRecord>& feats,
                       std::vector<int>& begin, std::vector<int>& count, std::string& err) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    if (!ctx->features_ready) {
        err = "features not resident";
        return false;
    }
    std::vector<Feature4> raw(size_t(ctx->resident_feats));
    CUDA_CHECK(cudaMemcpy(raw.data(), ctx->feats, sizeof(Feature4) * raw.size(),
                          cudaMemcpyDeviceToHost));
    feats.clear();
    feats.reserve(raw.size());
    for (const auto& f : raw) feats.push_back({f.x, f.y, f.label});
    begin.assign(size_t(ctx->pose_capacity) * 2, 0);
    count.assign(size_t(ctx->pose_capacity) * 2, 0);
    CUDA_CHECK(cudaMemcpy(begin.data(), ctx->feat_begin, sizeof(int) * begin.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(count.data(), ctx->feat_count, sizeof(int) * count.size(),
                          cudaMemcpyDeviceToHost));
    return true;
}

bool upload_template_dims(GpuContext* ctx, const int* widths, const int* heights, int len,
                          std::string& err, int model_slot, int frame_slot) {
    return upload_template_dims_impl(ctx, widths, heights, len, model_slot, frame_slot,
                                     nullptr, err);
}

bool upload_frame(GpuContext* ctx, const uint8_t* padded_bgr, int w, int h,
                  std::string& err) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    if (!padded_bgr) {
        err = "null frame";
        return false;
    }
    if (w != ctx->face[0][0].w || h != ctx->face[0][0].h) {
        err = "frame size mismatch with context";
        return false;
    }
    CUDA_CHECK(cudaMemcpy(ctx->face[0][0].bgr, padded_bgr, size_t(w) * h * 3,
                          cudaMemcpyHostToDevice));
    return true;
}

bool upload_frame_capacity(GpuContext* ctx, const uint8_t* padded_bgr, int w, int h,
                           std::string& err, int frame_slot) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    if (frame_slot != 0 && frame_slot != 1) {
        err = "frame slot out of range";
        return false;
    }
    if (!padded_bgr) {
        err = "null frame";
        return false;
    }
    if (w > ctx->face[frame_slot][0].w || h > ctx->face[frame_slot][0].h) {
        err = "frame size exceeds context capacity";
        return false;
    }
    // 内核只读写当前档区域，容量剩余部分不参与计算——位级行为与等档上下文一致
    ctx->cur_w[frame_slot] = w;
    ctx->cur_h[frame_slot] = h;
    CUDA_CHECK(cudaMemcpy(ctx->face[frame_slot][0].bgr, padded_bgr, size_t(w) * h * 3,
                          cudaMemcpyHostToDevice));
    return true;
}

bool upload_frame_capacity_async(GpuContext* ctx, const uint8_t* padded_bgr, int w, int h,
                                 void* stream_handle, std::string& err, int frame_slot) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    if (frame_slot != 0 && frame_slot != 1) {
        err = "frame slot out of range";
        return false;
    }
    if (!padded_bgr || !stream_handle) {
        err = "null frame or stream";
        return false;
    }
    if (w > ctx->face[frame_slot][0].w || h > ctx->face[frame_slot][0].h) {
        err = "frame size exceeds context capacity";
        return false;
    }
    // cur_w/h 仅在内核发射时由宿主侧读取（launch 参数），双工作面下逐 frame_slot 隔离
    ctx->cur_w[frame_slot] = w;
    ctx->cur_h[frame_slot] = h;
    CUDA_CHECK(cudaMemcpyAsync(ctx->face[frame_slot][0].bgr, padded_bgr, size_t(w) * h * 3,
                               cudaMemcpyHostToDevice, (cudaStream_t)stream_handle));
    return true;
}

bool upload_features_async(GpuContext* ctx, const FeatureRecord* feats, int feat_total,
                           const int* begin, const int* count, int layout_len, int poses,
                           int model_slot, void* stream_handle, std::string& err,
                           int frame_slot) {
    return upload_features_impl(ctx, feats, feat_total, begin, count, layout_len, poses,
                                model_slot, frame_slot, (cudaStream_t)stream_handle, err);
}

bool upload_template_dims_async(GpuContext* ctx, const int* widths, const int* heights, int len,
                                int model_slot, void* stream_handle, std::string& err,
                                int frame_slot) {
    return upload_template_dims_impl(ctx, widths, heights, len, model_slot, frame_slot,
                                     (cudaStream_t)stream_handle, err);
}

void* stream_create(std::string& err) {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        err = "stream create failed";
        return nullptr;
    }
    return stream;
}

void stream_destroy(void* stream) {
    if (stream) cudaStreamDestroy((cudaStream_t)stream);
}

void* event_create(std::string& err) {
    cudaEvent_t event = nullptr;
    if (cudaEventCreate(&event) != cudaSuccess) {
        err = "event create failed";
        return nullptr;
    }
    return event;
}

void event_destroy(void* event) {
    if (event) cudaEventDestroy((cudaEvent_t)event);
}

void event_record(void* event, void* stream) {
    if (event) cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
}

bool event_sync(void* event, std::string& err) {
    if (!event) {
        err = "null event";
        return false;
    }
    if (cudaEventSynchronize((cudaEvent_t)event) != cudaSuccess) {
        err = "event sync failed";
        return false;
    }
    return true;
}

double event_elapsed_ms(void* begin_ev, void* end_ev, bool& ok) {
    float ms = 0.f;
    ok = begin_ev && end_ev && cudaEventElapsedTime(&ms, (cudaEvent_t)begin_ev,
                                                    (cudaEvent_t)end_ev) == cudaSuccess;
    return ok ? double(ms) : 0.0;
}

void* pinned_alloc(size_t bytes, std::string& err) {
    void* ptr = nullptr;
    if (bytes == 0 || cudaMallocHost(&ptr, bytes) != cudaSuccess) {
        err = "pinned alloc failed";
        return nullptr;
    }
    return ptr;
}

void pinned_free(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

bool readback_frame(const GpuContext* ctx, uint8_t* padded_bgr, int w, int h,
                    std::string& err) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    if (!padded_bgr) {
        err = "null output";
        return false;
    }
    if (w != ctx->face[0][0].w || h != ctx->face[0][0].h) {
        err = "frame size mismatch with context";
        return false;
    }
    CUDA_CHECK(cudaMemcpy(padded_bgr, ctx->face[0][0].bgr, size_t(w) * h * 3,
                          cudaMemcpyDeviceToHost));
    return true;
}

bool lut_upload(const unsigned char host_lut[256], std::string& err) {
    if (!host_lut) {
        err = "null lut";
        return false;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(SIMILARITY_LUT, host_lut, 256));
    return true;
}

bool lut_readback(unsigned char host_lut[256], std::string& err) {
    if (!host_lut) {
        err = "null lut";
        return false;
    }
    CUDA_CHECK(cudaMemcpyFromSymbol(host_lut, SIMILARITY_LUT, 256));
    return true;
}

int context_pose_capacity(const GpuContext* ctx) { return ctx ? ctx->pose_capacity : 0; }

int context_candidate_capacity(const GpuContext* ctx) {
    return ctx ? ctx->candidate_capacity : 0;
}

bool device_info(std::string& name, int& major, int& minor, std::string& err) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    name = prop.name;
    major = prop.major;
    minor = prop.minor;
    return true;
}

}  // namespace vfshape::gpu
