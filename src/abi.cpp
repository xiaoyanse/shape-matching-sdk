#include "model.hpp"
#include "polygon.hpp"
#include "two_stage.hpp"
#include "audit.hpp"
#include "matching_profile.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

namespace {
std::mutex registry_mutex;

// ABI v3：句柄可为单级（旧资产）或两级容器模型。
struct ModelHolder {
    std::shared_ptr<vfshape::Model> single;            // 单级（旧格式资产）
    std::shared_ptr<vfshape::TwoStageModel> two_stage; // 两级容器（VF2S3）
    std::mutex mutex;
    vf_shape_match_trace last_trace{};                 // 最近一次 match/batch 诊断快照
    ModelHolder() { last_trace.struct_size = sizeof(vf_shape_match_trace); }
};
std::unordered_map<uint64_t, std::shared_ptr<ModelHolder>> registry;
uint64_t next_handle = 1;
constexpr size_t max_handles = 128;

void copy_error(char* dest, size_t dest_size, const char* message) {
    const auto length = std::min(dest_size - 1, std::strlen(message));
    std::memcpy(dest, message, length); dest[length] = '\0';
}

template<class Function>
int32_t boundary(char* error, uint32_t capacity, Function&& body) noexcept {
    if (error && capacity) error[0] = '\0';
    const uint64_t scope = vfshape::begin_error_scope();
    try { body(); return 0; }
    catch (const std::exception& ex) {
        vfshape::finish_error_scope(scope);
        if (error && capacity) copy_error(error, capacity, ex.what());
        return 1;
    } catch (...) {
        vfshape::finish_error_scope(scope);
        if (error && capacity) copy_error(error, capacity, "unknown native matching failure");
        return 2;
    }
}

// 诊断快照落句柄档案（单级无两级编排：仅最终候选数有效，其余恒 0）
void store_two_stage_trace(ModelHolder& holder, const vfshape::MatchTraceSnapshot& snap) {
    holder.last_trace.struct_size = sizeof(vf_shape_match_trace);
    holder.last_trace.coarse_candidates = uint32_t(std::max(0, snap.coarse_candidates));
    holder.last_trace.windows_searched = uint32_t(std::max(0, snap.windows_searched));
    holder.last_trace.final_candidates = uint32_t(std::max(0, snap.final_candidates));
    std::copy_n(snap.window_best.begin(), 8, holder.last_trace.window_best_scores);
    holder.last_trace.coarse_ms = snap.coarse_ms;
    holder.last_trace.coarse_resize_ms = snap.coarse_resize_ms;
    holder.last_trace.windows_ms = snap.windows_ms;
    holder.last_trace.finish_ms = snap.finish_ms;
}
void store_single_trace(ModelHolder& holder, size_t final_count) {
    const uint32_t struct_size = holder.last_trace.struct_size;
    holder.last_trace = vf_shape_match_trace{};
    holder.last_trace.struct_size = struct_size;
    holder.last_trace.final_candidates = uint32_t(final_count);
}

// 批量备帧累计预算按填充后像素计价：每帧常驻约 5.75B/px（两层源图金字塔
// 3.75B/px、末层 8 方向线性内存 2B/px），匹配期惰性展开响应图再增约一半；
// item_count 上限 16 组同备 4096² 帧峰值可超 2GB。上限取 8 张 4096² 等效
// （填充后 4160²）把峰值压到约 1GB，超限组整组拒绝而非整批失败；两级项
// 不经共享帧不受此限。
constexpr uint64_t kMaxBatchPreparedPixels = 8ULL * 4160 * 4160;

uint64_t padded_frame_pixels(int width, int height) {
    const auto padded = [](int edge) {
        return uint64_t((edge + 2 * vfshape::PreparedFrame::padding + 15) / 16 * 16);
    };
    return padded(width) * padded(height);
}
std::shared_ptr<ModelHolder> get(uint64_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto found = registry.find(handle);
    vfshape::require_coded(found != registry.end(), VF_SHAPE_HANDLE_INVALID,
                           "model handle is closed or unknown", double(handle));
    return found->second;
}
void put(std::shared_ptr<ModelHolder> model, uint64_t* handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    vfshape::require_coded(registry.size() < max_handles && next_handle != 0,
                           VF_SHAPE_HANDLE_BUDGET, "native handle budget exceeded",
                           double(registry.size()), double(max_handles));
    *handle = next_handle++; registry.emplace(*handle, std::move(model));
}
}

uint32_t vf_shape_abi_version() { return 3; }  // v3: 两级编排与几何精修全量下沉；训练默认两级打包
int32_t vf_shape_train(const uint8_t* image, uint64_t image_size, int32_t width, int32_t height,
    uint64_t stride, const vf_shape_train_config* config, uint64_t* handle, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(handle && config, "missing training output/config"); *handle = 0;
        const auto frame = vfshape::checked_image(image, image_size, width, height, stride);
        auto holder = std::make_shared<ModelHolder>();
        holder->two_stage = vfshape::train_two_stage(frame, *config);
        put(std::move(holder), handle);
    });
}
int32_t vf_shape_load(const uint8_t* bytes, uint64_t size, uint64_t* handle, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(handle, "missing model handle output"); *handle = 0;
        auto holder = std::make_shared<ModelHolder>();
        holder->two_stage = vfshape::decode_two_stage(bytes, size);
        if (!holder->two_stage) holder->single = vfshape::decode(bytes, size);
        vfshape::require_coded(holder->single || holder->two_stage,
                               VF_SHAPE_MODEL_FORMAT,
                               "model is neither two-stage container nor single-level format");
        put(std::move(holder), handle);
    });
}
int32_t vf_shape_export(uint64_t handle, uint8_t* bytes, uint64_t capacity, uint64_t* size,
    char* error, uint32_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        vfshape::require(size, "missing model size output");
        auto model = get(handle); std::lock_guard<std::mutex> lock(model->mutex);
        std::vector<uint8_t> encoded;
        if (model->two_stage)
            encoded = vfshape::encode_two_stage(*model->two_stage);
        else
            encoded = vfshape::encode(*model->single);
        *size = encoded.size();
        if (!bytes && capacity == 0) return;
        vfshape::require(bytes && capacity >= encoded.size(), "model output buffer too small");
        std::memcpy(bytes, encoded.data(), encoded.size());
    });
}
int32_t vf_shape_match(uint64_t handle, const uint8_t* image, uint64_t image_size, int32_t width,
    int32_t height, uint64_t stride, double threshold, vf_shape_candidate* candidates,
    uint32_t capacity, uint32_t* count, char* error, uint32_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        vfshape::require(count, "missing candidate count output");
        *count = 0;
        vfshape::require_coded(candidates && capacity >= 2 && capacity <= 256,
                               VF_SHAPE_CAPACITY_INVALID, "candidate capacity must be 2..256",
                               double(capacity), 2.0, 256.0);
        const auto frame = vfshape::checked_image(image, image_size, width, height, stride);
        auto model = get(handle); std::lock_guard<std::mutex> lock(model->mutex);
        std::vector<vf_shape_candidate> found;
        if (model->two_stage) {
            vfshape::MatchTraceSnapshot snap;
            found = vfshape::match_two_stage(*model->two_stage, frame, threshold,
                                             int(capacity), &snap);
            store_two_stage_trace(*model, snap);
        } else {
            found = vfshape::match(*model->single, frame, threshold);
            store_single_trace(*model, found.size());
        }
        *count = uint32_t(std::min(size_t(capacity), found.size()));
        std::copy_n(found.begin(), *count, candidates);
    });
}
int32_t vf_shape_metadata(uint64_t handle, vf_shape_train_config* config, int32_t* width,
    int32_t* height, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(config && width && height && config->struct_size == sizeof(*config),
                         "invalid metadata output ABI");
        auto model = get(handle);
        std::lock_guard<std::mutex> lock(model->mutex);
        if (model->two_stage) {
            *config = model->two_stage->base_config;
            // 参考尺寸须与 config 同处原图坐标系；旧容器 ref 字段为 0，回退模板灰图尺寸
            *width = model->two_stage->ref_width > 0
                         ? model->two_stage->ref_width : model->two_stage->tmpl_gray.cols;
            *height = model->two_stage->ref_height > 0
                          ? model->two_stage->ref_height : model->two_stage->tmpl_gray.rows;
        } else {
            *config = model->single->config; *width = model->single->width; *height = model->single->height;
        }
    });
}
int32_t vf_shape_match_batch(const uint8_t* image, uint64_t image_size, int32_t width,
    int32_t height, uint64_t stride, vf_shape_batch_item* items, uint32_t item_count,
    uint32_t item_size, char* error, uint32_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        vfshape::require(items && item_count >= 1 && item_count <= 16 &&
            item_size == sizeof(vf_shape_batch_item), "invalid batch output ABI or count");
        for (uint32_t n = 0; n < item_count; ++n) {
            items[n].count = 0; items[n].status = 1; items[n].error[0] = '\0';
        }
        const auto image_view = vfshape::checked_image(image, image_size, width, height, stride);
        std::vector<std::shared_ptr<ModelHolder>> models(item_count);
        bool any_valid = false;
        for (uint32_t n = 0; n < item_count; ++n) {
            auto& item = items[n];
            item.status = boundary(item.error, sizeof(item.error), [&] {
                vfshape::require_coded(item.candidates && item.capacity >= 2 && item.capacity <= 256,
                    VF_SHAPE_CAPACITY_INVALID, "candidate capacity must be 2..256",
                    double(item.capacity), 2.0, 256.0);
                vfshape::require_coded(std::isfinite(item.threshold) && item.threshold > 0 &&
                    item.threshold <= 100, VF_SHAPE_THRESHOLD_INVALID,
                    "invalid similarity threshold", item.threshold);
                auto model = get(item.handle);
                // 搜索预算仅约束单级共享帧匹配；两级匹配内部自管缩图与窗口预算
                if (model->single)
                    vfshape::validate_search_budget(*model->single, width, height);
                models[n] = std::move(model);
                any_valid = true;
            });
        }
        if (!any_valid) return;
        // 单级帧量化与模板特征选择必须同参：按生效 detector 参数分组共帧，
        // 跨参共享 PreparedFrame 会触发 vendor 同参断言；两级匹配内部自行缩图，不共享。
        std::map<std::tuple<int, float, float>, vfshape::PreparedFrame> frames;
        std::set<std::tuple<int, float, float>> failed_groups;
        auto fail_group = [&](const std::tuple<int, float, float>& key, const char* message,
                              int32_t status = 1) {
            failed_groups.insert(key);
            for (uint32_t m = 0; m < item_count; ++m) {
                if (!models[m] || !models[m]->single) continue;
                const auto& detector = models[m]->single->detector;
                if (std::make_tuple(detector.tuned_features, detector.tuned_weak,
                                    detector.tuned_strong) != key)
                    continue;
                items[m].status = status;
                copy_error(items[m].error, sizeof(items[m].error), message);
            }
        };
        const auto frame_pixels = padded_frame_pixels(width, height);
        uint64_t prepared_pixels = 0;
        for (uint32_t n = 0; n < item_count; ++n) {
            if (!models[n] || !models[n]->single) continue;
            const auto& detector = models[n]->single->detector;
            auto key = std::make_tuple(detector.tuned_features, detector.tuned_weak,
                                       detector.tuned_strong);
            if (frames.count(key) || failed_groups.count(key)) continue;
            if (prepared_pixels + frame_pixels > kMaxBatchPreparedPixels) {
                vfshape::record_error(VF_SHAPE_PIXEL_BUDGET,
                                      {double(kMaxBatchPreparedPixels),
                                       double(prepared_pixels), double(frame_pixels)});
                const auto message = "batch prepared-frame pixel budget exceeded: budget "
                    + std::to_string(kMaxBatchPreparedPixels) + " pixels, prepared "
                    + std::to_string(prepared_pixels) + " pixels, frame "
                    + std::to_string(frame_pixels) + " pixels";
                fail_group(key, message.c_str());
                continue;
            }
            try {
                // emplace 对已存在键仍会求值实参——先查再插，保证同参组只备一次帧
                frames.emplace(key, vfshape::prepare_frame(image_view, &detector));
            } catch (const std::exception& ex) {
                fail_group(key, ex.what());
                continue;
            } catch (...) {
                fail_group(key, "unknown native frame preparation failure", 2);
                continue;
            }
            prepared_pixels += frame_pixels;
        }
        for (uint32_t n = 0; n < item_count; ++n) {
            if (!models[n]) continue;
            auto& item = items[n];
            if (models[n]->single) {
                const auto& detector = models[n]->single->detector;
                if (failed_groups.count(std::make_tuple(detector.tuned_features,
                                                        detector.tuned_weak,
                                                        detector.tuned_strong)))
                    continue;  // status/error 已在备帧隔离时标记
            }
            item.status = boundary(item.error, sizeof(item.error), [&] {
                std::lock_guard<std::mutex> lock(models[n]->mutex);
                std::vector<vf_shape_candidate> found;
                if (models[n]->two_stage) {
                    vfshape::MatchTraceSnapshot snap;
                    found = vfshape::match_two_stage(*models[n]->two_stage, image_view,
                                                     item.threshold, int(item.capacity), &snap);
                    store_two_stage_trace(*models[n], snap);
                } else {
                    const auto& detector = models[n]->single->detector;
                    const auto& frame = frames.find(std::make_tuple(
                        detector.tuned_features, detector.tuned_weak, detector.tuned_strong))->second;
                    found = vfshape::match_prepared(*models[n]->single, frame, item.threshold);
                    store_single_trace(*models[n], found.size());
                }
                item.count = uint32_t(std::min(size_t(item.capacity), found.size()));
                std::copy_n(found.begin(), item.count, item.candidates);
            });
        }
    });
}
int32_t vf_shape_close(uint64_t handle, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        std::lock_guard<std::mutex> lock(registry_mutex);
        registry.erase(handle);
    });
}
void vf_shape_last_error_info(int32_t* code, double params[6], int32_t* param_count) {
    if (code) *code = vfshape::g_last_error.code;
    if (param_count) *param_count = vfshape::g_last_error.param_count;
    if (params)
        std::copy_n(vfshape::g_last_error.params, 6, params);
}
// 预算报告 v1 布局尺寸（polygon_fill_ratio 尾部追加前）：新头部 sizeof=72，
// 旧头部消费方仍传 64；入口按 struct_size 判别，只回写调用方声明长度的前缀
constexpr uint32_t kBudgetReportV1Size = 64;
static_assert(sizeof(vf_shape_budget_report) ==
              kBudgetReportV1Size + sizeof(double));

void require_budget_report_size(const vf_shape_budget_report& out) {
    vfshape::require(out.struct_size == sizeof(vf_shape_budget_report) ||
                         out.struct_size == kBudgetReportV1Size,
                     "invalid budget report output ABI");
}

void copy_budget_report(vf_shape_budget_report& out, const vf_shape_budget_report& report) {
    std::memcpy(&out, &report, std::min(sizeof(out), size_t(out.struct_size)));
}

int32_t vf_shape_estimate_training(const vf_shape_train_config* config, int32_t image_w,
    int32_t image_h, vf_shape_budget_report* out, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(config && out, "missing budget report output/config");
        require_budget_report_size(*out);
        copy_budget_report(*out, vfshape::estimate_training(*config, image_w, image_h));
    });
}
int32_t vf_shape_estimate_training_ex(const vf_shape_train_config* config, int32_t image_w,
    int32_t image_h, const vf_shape_polygon_train_params* params,
    vf_shape_budget_report* out, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(config && out, "missing budget report output/config");
        vfshape::require(config->struct_size == sizeof(*config),
                         "invalid training ABI structure size");
        require_budget_report_size(*out);
        copy_budget_report(*out, vfshape::estimate_training_ex(*config, image_w, image_h,
                                                               params));
    });
}
int32_t vf_shape_train_polygon(const uint8_t* image, uint64_t image_size, int32_t width,
    int32_t height, uint64_t stride, const vf_shape_polygon_train_params* params,
    uint64_t* handle, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(handle && params, "missing polygon training output/params");
        *handle = 0;
        const auto frame = vfshape::checked_image(image, image_size, width, height, stride);
        const auto shape = vfshape::validate_polygon(*params, width, height);
        const auto config = vfshape::config_from_polygon(*params, shape);
        auto holder = std::make_shared<ModelHolder>();
        holder->two_stage = vfshape::train_two_stage(frame, config, &shape);
        put(std::move(holder), handle);
    });
}
// 建议条数为返回值（0=无建议非错误），失败走负值分层：-1 可报告 / -2 未分类
int32_t suggest_boundary(const vf_shape_suggest_params* params, const uint8_t* image,
    uint64_t image_size, int32_t width, int32_t height, uint64_t stride,
    const int32_t* roi_hint, uint32_t max_suggestions, uint32_t max_vertices_per,
    vf_shape_vertex* out_vertices, uint32_t* out_vertex_counts,
    char* error, uint32_t error_capacity) {
    if (error && error_capacity) error[0] = '\0';
    const uint64_t scope = vfshape::begin_error_scope();
    try {
        vfshape::require(out_vertices && out_vertex_counts, "missing suggestion output buffers");
        const auto frame = vfshape::checked_image(image, image_size, width, height, stride);
        return int32_t(vfshape::suggest_contours(frame, roi_hint, max_suggestions,
                                                 max_vertices_per, params, out_vertices,
                                                 out_vertex_counts));
    } catch (const std::exception& ex) {
        vfshape::finish_error_scope(scope);
        if (error && error_capacity) copy_error(error, error_capacity, ex.what());
        return -1;
    } catch (...) {
        vfshape::finish_error_scope(scope);
        if (error && error_capacity)
            copy_error(error, error_capacity, "unknown native matching failure");
        return -2;
    }
}
int32_t vf_shape_suggest_contours(const uint8_t* image, uint64_t image_size, int32_t width,
    int32_t height, uint64_t stride, const int32_t* roi_hint, uint32_t max_suggestions,
    uint32_t max_vertices_per, vf_shape_vertex* out_vertices, uint32_t* out_vertex_counts,
    char* error, uint32_t error_capacity) {
    return suggest_boundary(nullptr, image, image_size, width, height, stride, roi_hint,
                            max_suggestions, max_vertices_per, out_vertices,
                            out_vertex_counts, error, error_capacity);
}
int32_t vf_shape_suggest_contours_ex(const uint8_t* image, uint64_t image_size, int32_t width,
    int32_t height, uint64_t stride, const int32_t* roi_hint, uint32_t max_suggestions,
    uint32_t max_vertices_per, const vf_shape_suggest_params* params,
    vf_shape_vertex* out_vertices, uint32_t* out_vertex_counts,
    char* error, uint32_t error_capacity) {
    return suggest_boundary(params, image, image_size, width, height, stride, roi_hint,
                            max_suggestions, max_vertices_per, out_vertices,
                            out_vertex_counts, error, error_capacity);
}
int32_t vf_shape_polygon_metadata(uint64_t handle, vf_shape_vertex* outline, uint32_t capacity,
    uint32_t* outline_count, uint32_t* hole_count, char* error, uint32_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        vfshape::require(outline_count && hole_count, "missing polygon metadata output");
        auto model = get(handle);
        std::lock_guard<std::mutex> lock(model->mutex);
        std::vector<cv::Point2d> points;
        uint32_t holes = 0;
        if (model->two_stage && model->two_stage->polygon) {
            const auto& flat = model->two_stage->polygon->outline;
            points.reserve(flat.size() / 2);
            for (size_t i = 0; i + 1 < flat.size(); i += 2)
                points.push_back({flat[i], flat[i + 1]});
            holes = uint32_t(model->two_stage->polygon->holes.size());
        } else {
            // 矩形资产（含旧格式与单级资产）回退 ROI 四角：左上、右上、右下、左下
            const auto& c = model->two_stage ? model->two_stage->base_config
                                             : model->single->config;
            points = {{double(c.roi_x), double(c.roi_y)},
                      {double(c.roi_x + c.roi_width), double(c.roi_y)},
                      {double(c.roi_x + c.roi_width), double(c.roi_y + c.roi_height)},
                      {double(c.roi_x), double(c.roi_y + c.roi_height)}};
        }
        *outline_count = uint32_t(points.size());
        *hole_count = holes;
        if (!outline && capacity == 0) return;
        vfshape::require_coded(capacity >= points.size(), VF_SHAPE_CAPACITY_INVALID,
                               "polygon metadata buffer too small", double(capacity),
                               double(points.size()));
        for (size_t i = 0; i < points.size(); ++i) {
            outline[i].x = points[i].x;
            outline[i].y = points[i].y;
        }
    });
}
int32_t vf_shape_last_match_info(uint64_t handle, vf_shape_match_trace* out,
    char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(out, "missing match trace output");
        vfshape::require(out->struct_size == sizeof(*out), "invalid match trace output ABI");
        auto model = get(handle);
        std::lock_guard<std::mutex> lock(model->mutex);
        *out = model->last_trace;
    });
}
int32_t vf_shape_model_health(uint64_t handle, vf_shape_health_report* out,
    char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(out, "missing health report output");
        vfshape::require(out->struct_size == sizeof(*out), "invalid health report output ABI");
        auto model = get(handle);
        std::lock_guard<std::mutex> lock(model->mutex);
        *out = vfshape::model_health(model->single.get(), model->two_stage.get());
    });
}
int32_t vf_shape_audit_discriminability(uint64_t handle,
    const uint8_t* const* pos_images, int32_t pos_width, int32_t pos_height,
    uint64_t pos_stride, uint32_t pos_count,
    const uint8_t* const* neg_images, int32_t neg_width, int32_t neg_height,
    uint64_t neg_stride, uint32_t neg_count,
    vf_shape_audit_report* out, char* error, uint32_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        vfshape::require(out && pos_images && pos_count >= 1,
                         "missing audit report output or positive images");
        vfshape::require(neg_count == 0 || neg_images, "missing negative image array");
        vfshape::require(out->struct_size == sizeof(*out), "invalid audit report output ABI");
        auto model = get(handle);
        std::lock_guard<std::mutex> lock(model->mutex);
        *out = vfshape::audit_discriminability(
            model->single.get(), model->two_stage.get(),
            pos_images, pos_width, pos_height, pos_stride, pos_count,
            neg_images, neg_width, neg_height, neg_stride, neg_count);
    });
}
