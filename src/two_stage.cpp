// 两级匹配、分组模型训练和版本化容器。
// 容器格式字节级兼容（VF2S3）；单级旧资产向后兼容。

#include "two_stage.hpp"
#include "matching_profile.hpp"
#include "coarse_image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vfshape {
namespace {

constexpr char kMagic[] = "VF2S3";
constexpr size_t kMagicLen = 5;
constexpr size_t kLenWidth = 8;

uint64_t read_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
    return v;
}
void write_u64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}

// F09 模板段格式合同=仅 PNG：本方编码器只产 PNG（encode_two_stage 用
// cv::imencode(".png")），收紧无合法资产受损。imdecode 还可解 BMP/PGM/JPEG
// 等，若放行非 PNG 字节，尺寸限制退化为解码后补救、解码分配无界——故不足
// 24 字节/签名不符/首块非 IHDR/宽或高（u32 BE，偏移 16/20）>4096 一律在
// 解码前拒绝。文案统一 "template image header ..." 系，与解码后检查可区分
void require_png_template(const std::vector<uint8_t>& png) {
    static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    require(png.size() >= 24, "template image header too short for PNG");
    require(std::memcmp(png.data(), signature, sizeof(signature)) == 0,
            "template image header signature is not PNG");
    require(std::memcmp(png.data() + 12, "IHDR", 4) == 0,
            "template image header first chunk is not IHDR");
    auto u32_be = [&png](size_t off) {
        return (uint64_t(png[off]) << 24) | (uint64_t(png[off + 1]) << 16) |
               (uint64_t(png[off + 2]) << 8) | uint64_t(png[off + 3]);
    };
    require(u32_be(16) <= uint64_t(max_edge) && u32_be(20) <= uint64_t(max_edge),
            "template image header exceeds edge budget");
}

struct Meta {
    int k = 8;
    std::vector<GroupSpec> groups;
    // F01 原始终久化键：旧容器无 ref/cfg → 派生回退（fines 首组配置冒充）
    bool has_ref = false;
    int ref_w = 0, ref_h = 0;
    bool has_cfg = false;
    vf_shape_train_config cfg{};
};

Meta parse_meta(const std::string& json) {
    cv::FileStorage storage(json, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                     cv::FileStorage::FORMAT_JSON);
    require(storage.isOpened(), "invalid container metadata");
    const auto factor = storage["k"];
    require(factor.isInt(), "coarse factor must be an integer");
    Meta meta;
    meta.k = int(factor);
    require(meta.k == 2 || meta.k == 4 || meta.k == 8, "invalid coarse factor");
    const auto groups = storage["groups"], biases = storage["group_bias"];
    require(groups.isSeq() && biases.isSeq() && groups.size() == biases.size() &&
                groups.size() > 0 && groups.size() <= 64, "invalid group metadata");
    auto number = [](const cv::FileNode& node) {
        require(node.isInt() || node.isReal(), "group metadata must be numeric");
        const double value = double(node);
        require(std::isfinite(value), "group metadata must be finite");
        return value;
    };
    auto integer = [&number](const cv::FileNode& node) {
        const double value = number(node);
        require(value == std::floor(value) && std::abs(value) <= 2147483647.0,
                "container metadata must be a 32-bit integer");
        return value;
    };
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto bounds = groups[int(i)], bias = biases[int(i)];
        require(bounds.isSeq() && bounds.size() == 2 && bias.isSeq() && bias.size() == 2,
                "invalid group bounds or anchor");
        GroupSpec group{number(bounds[0]), number(bounds[1]), number(bias[0]), number(bias[1])};
        require(group.lo >= -180 && group.hi <= 180 && group.lo <= group.hi,
                "invalid group angle range");
        require(std::abs(group.bias_x) <= max_edge && std::abs(group.bias_y) <= max_edge,
                "invalid group anchor");
        meta.groups.push_back(group);
    }
    // F01：可选原始配置键（旧容器缺失 → 派生回退；两键必须成对出现）
    const auto ref = storage["ref"], cfg = storage["cfg"];
    require(ref.isNone() == cfg.isNone(), "reference size and config must appear together");
    if (!ref.isNone()) {
        require(ref.isSeq() && ref.size() == 2, "invalid reference size metadata");
        const double rw = integer(ref[0]), rh = integer(ref[1]);
        require(rw >= 24 && rw <= max_edge && rh >= 24 && rh <= max_edge,
                "invalid reference size metadata");
        meta.ref_w = int(rw);
        meta.ref_h = int(rh);
        meta.has_ref = true;
    }
    if (!cfg.isNone()) {
        require(cfg.isSeq() && cfg.size() == 13, "invalid training config metadata");
        vf_shape_train_config c{};
        c.struct_size = sizeof(c);
        c.roi_x = int(integer(cfg[0]));
        c.roi_y = int(integer(cfg[1]));
        c.roi_width = int(integer(cfg[2]));
        c.roi_height = int(integer(cfg[3]));
        c.angle_min = number(cfg[4]);
        c.angle_max = number(cfg[5]);
        c.angle_step = number(cfg[6]);
        c.scale_min = number(cfg[7]);
        c.scale_max = number(cfg[8]);
        c.scale_step = number(cfg[9]);
        c.weak_th = number(cfg[10]);
        c.strong_th = number(cfg[11]);
        c.feature_count = int(integer(cfg[12]));
        meta.cfg = c;
        meta.has_cfg = true;
    }
    return meta;
}

std::string dump_meta(const TwoStageModel& model) {
    std::ostringstream out;
    out << std::setprecision(17);  // double 往返无损（组界/偏置/配置同为 17 位）
    out << "{\"k\":" << model.k << ",\"groups\":[";
    for (size_t i = 0; i < model.groups.size(); ++i) {
        if (i) out << ",";
        out << "[" << model.groups[i].lo << "," << model.groups[i].hi << "]";
    }
    out << "],\"group_bias\":[";
    for (size_t i = 0; i < model.groups.size(); ++i) {
        if (i) out << ",";
        out << "[" << model.groups[i].bias_x << "," << model.groups[i].bias_y << "]";
    }
    out << "]";
    // F01：原始终久化——参考图尺寸 + 原始训练配置（零值默认语义原样保存）。
    // 仅 ref 有效（新产线/新容器往返）时写入；旧容器回退模型无原始值，不落键
    if (model.ref_width > 0 && model.ref_height > 0) {
        const auto& c = model.base_config;
        out << ",\"ref\":[" << model.ref_width << "," << model.ref_height << "]";
        out << ",\"cfg\":[" << c.roi_x << "," << c.roi_y << "," << c.roi_width << ","
            << c.roi_height << "," << c.angle_min << "," << c.angle_max << "," << c.angle_step
            << "," << c.scale_min << "," << c.scale_max << "," << c.scale_step << ","
            << c.weak_th << "," << c.strong_th << "," << c.feature_count << "]";
    }
    out << "}";
    return out.str();
}

int effective_k(int roi_w, int roi_h) {
    // Python 对齐：阈值 56px（K8→K4→K2 降档）
    const int longest = std::max(roi_w, roi_h);
    for (int k : {8, 4, 2})
        if (longest >= 56 * k) return k;
    return 2;
}

// 恒等（或指定角）自匹配测锚点偏置：Python _measure_anchor_bias 对齐
void measure_bias(Model& model, const cv::Mat& image_bgr,
                  const cv::Rect& roi, double angle_deg, double* bx, double* by) {
    // 注意：target 必须独立分配——`cv::Mat target = image_bgr` 浅拷贝共享数据，
    // warpAffine 的 src/dst 同 Mat 是未定义行为（实测旋转图损坏 74% 像素差异，
    // 组偏置被测成 (19,-14) 级离谱值）。
    // 性能：旋转与匹配限制在 ROI 邻域窗（目标位置已知=ROI 中心±模板半径），
    // 全图 match ~400ms → 窗口 ~30ms（偏置测量 4 次/训练）。
    const double cx = roi.x + roi.width / 2.0, cy = roi.y + roi.height / 2.0;
    const double radius = std::hypot(double(roi.width), double(roi.height)) / 2.0 + 64.0;
    const int wx0 = int(std::max(0.0, cx - radius)), wy0 = int(std::max(0.0, cy - radius));
    const int wx1 = int(std::min(double(image_bgr.cols), cx + radius + 1.0));
    const int wy1 = int(std::min(double(image_bgr.rows), cy + radius + 1.0));
    if (wx1 - wx0 < 64 || wy1 - wy0 < 64) {
        *bx = 0;
        *by = 0;
        return;
    }
    cv::Mat patch = image_bgr(cv::Rect(wx0, wy0, wx1 - wx0, wy1 - wy0));
    cv::Mat target;
    const double pcx = cx - wx0, pcy = cy - wy0;  // ROI 中心的窗口系坐标
    if (std::abs(angle_deg) > 1e-9) {
        auto m = cv::getRotationMatrix2D(cv::Point2d(pcx, pcy), angle_deg, 1.0);
        cv::warpAffine(patch, target, m, patch.size());
    } else {
        target = patch;
    }
    // F06 去异常化：无 try/catch。空候选本身是预期降级信号（零偏置）；
    // 真异常（bad_alloc/OpenCV 运行时/预算合同）自然穿透训练报错
    auto cands = match(model, target, 50.0);
    if (cands.empty()) {
        // 降级语义：锚点校准依赖 ROI 自匹配，低对比/弱特征场景落空是预期路径
        *bx = 0;
        *by = 0;
        return;
    }
    const auto& best = *std::max_element(
        cands.begin(), cands.end(),
        [](const vf_shape_candidate& a, const vf_shape_candidate& b) {
            return a.score < b.score;
        });
    *bx = best.center[0] - pcx;
    *by = best.center[1] - pcy;
}

}  // namespace


// —— 两级训练（Python train_packaged 对齐）——
std::shared_ptr<TwoStageModel> train_two_stage(
    const cv::Mat& image_bgr, const vf_shape_train_config& config) {
    const int w = image_bgr.cols, h = image_bgr.rows;
    validate_config(config, w, h);
    const cv::Rect roi(config.roi_x, config.roi_y, config.roi_width, config.roi_height);

    // 角度分组：group_span=16°（8° 步长+8° 余量），重叠 8°
    const double fine_step = std::min(config.angle_step, 0.5);
    const double total = config.angle_max - config.angle_min;
    const double group_span = kCoarseAngleStep * 2;
    const int n_groups =
        (total > group_span) ? int(std::ceil(total / group_span)) : 1;

    auto model = std::make_shared<TwoStageModel>();
    model->base_config = config;
    // R01 对齐：cfg 三参持久化实际生效值（默认零在训练侧物化），与单级 codec
    // 的回填同口径——新容器不携带零字段，解码侧按持久化值严格校验不再冲突
    const auto tuned = validate_detector_params(config.feature_count, config.weak_th,
                                                config.strong_th);
    model->base_config.feature_count = tuned.features;
    model->base_config.weak_th = tuned.weak;
    model->base_config.strong_th = tuned.strong;
    model->ref_width = w;   // F01：原始参考图尺寸（容器 ref 键）
    model->ref_height = h;
    model->groups.resize(n_groups);

    auto train_fine = [&](double lo, double hi) {
        vf_shape_train_config fc = config;
        fc.angle_min = lo;
        fc.angle_max = hi;
        fc.angle_step = fine_step;
        fc.scale_min = 1.0;
        fc.scale_max = 1.0;
        fc.scale_step = 1.0;
        // 相邻组来自同一张参考图，重叠角度复用完全相同的特征。
        return train(image_bgr, fc, model->fines.empty() ? nullptr : model->fines.back().get());
    };

    if (n_groups <= 1) {
        model->fines.push_back(train_fine(config.angle_min, config.angle_max));
        model->groups[0] = {config.angle_min, config.angle_max, 0, 0};
    } else {
        const double overlap = kCoarseAngleStep;
        const double stride = total / n_groups;
        for (int i = 0; i < n_groups; ++i) {
            double lo = config.angle_min + i * stride - (i ? overlap : 0);
            double hi = config.angle_min + (i + 1) * stride + (i < n_groups - 1 ? overlap : 0);
            lo = std::max(config.angle_min, lo);
            hi = std::min(config.angle_max, hi);
            model->fines.push_back(train_fine(lo, hi));
            model->groups[i] = {lo, hi, 0, 0};
        }
    }

    // 粗层：K 逐档重试（Python 对齐：56 阈值 + 失败显式抛错）
    const int k0 = effective_k(config.roi_width, config.roi_height);
    const int coarse_steps = std::min(
        3, std::max(1, int(std::lround((config.scale_max - config.scale_min) /
                                        std::max(0.05, config.scale_step / 2))) + 1));
    bool coarse_ok = false;
    std::string coarse_failure;
    for (int k : {8, 4, 2}) {
        if (k > k0) continue;
        cv::Mat small;
        resize_coarse_image(image_bgr, small, k);
        vf_shape_train_config cc = config;
        // 缩图尺寸同为 floor 整除（coarse_image.cpp），ROI 同口径映射：
        // 原点允许 0（整图 ROI），宽高对实际缩图裁限
        cc.roi_x = config.roi_x / k;
        cc.roi_y = config.roi_y / k;
        cc.roi_width = std::min(config.roi_width / k, small.cols - cc.roi_x);
        cc.roi_height = std::min(config.roi_height / k, small.rows - cc.roi_y);
        cc.angle_step = kCoarseAngleStep;
        cc.scale_step = std::max(0.05, (config.scale_max - config.scale_min) /
                                            std::max(1, coarse_steps - 1));
        if (cc.roi_width < 8 || cc.roi_height < 8) {
            coarse_failure = "K=" + std::to_string(k) + ": coarse ROI " +
                             std::to_string(cc.roi_width) + "x" + std::to_string(cc.roi_height) +
                             " below 8px minimum";
            continue;
        }
        try {
            model->coarse = train(small, cc);
            model->k = k;
            coarse_ok = true;
            break;
        } catch (const std::invalid_argument& ex) {
            // F06 异常精确分类：仅输入合同类失败（require 抛 invalid_argument：
            // 特征不足/几何/预算）可换档重试，各档留真实失败原因；
            // bad_alloc/OpenCV 运行时等其他异常立即重抛穿透，不吞成"换档可解决"
            coarse_failure = "K=" + std::to_string(k) + ": " + ex.what();
        }
    }
    if (!coarse_ok)
        throw std::runtime_error("coarse model training failed at all K levels (" +
                                 coarse_failure + ")");

    // 模板灰度 + B7 点
    cv::Mat roi_bgr = image_bgr(roi);
    cv::cvtColor(roi_bgr, model->tmpl_gray, cv::COLOR_BGR2GRAY);
    model->b7 = extract_b7_points(model->tmpl_gray);

    // 组级偏置：每组在其中心角合成旋转图上测（Python 对齐）
    for (size_t i = 0; i < model->groups.size(); ++i) {
        const double mid = (model->groups[i].lo + model->groups[i].hi) / 2.0;
        measure_bias(*model->fines[i], image_bgr, roi, mid,
                     &model->groups[i].bias_x, &model->groups[i].bias_y);
    }
    return model;
}

// —— 容器编解码 ——
std::vector<uint8_t> encode_two_stage(const TwoStageModel& model) {
    // 段：fines 导出 bytes ×n + coarse + tmpl_png + meta
    // F09：增量预算——每生成一段即累计，中途超 32MiB 立即抛（不等全部段生成
    // 完，缩短超限模型的编码期驻留）；合同与 decode 侧同一口径
    uint64_t total = kMagicLen + kLenWidth * (uint64_t(model.fines.size()) + 4);
    auto accrue = [&total](uint64_t bytes) {
        total += bytes;
        require(total <= max_model_bytes, "model container exceeds size budget");
    };
    std::vector<std::vector<uint8_t>> fines_bytes;
    for (auto& f : model.fines) {
        fines_bytes.push_back(encode(*f));
        accrue(fines_bytes.back().size());
    }
    std::vector<uint8_t> coarse_bytes = encode(*model.coarse);
    accrue(coarse_bytes.size());
    std::vector<uint8_t> tmpl_bytes;
    cv::imencode(".png", model.tmpl_gray, tmpl_bytes);
    accrue(tmpl_bytes.size());
    const std::string meta = dump_meta(model);
    accrue(meta.size());

    std::vector<uint8_t> out(kMagic, kMagic + kMagicLen);
    write_u64(out, model.fines.size());
    // 长度前缀：fines... + coarse + tmpl + meta
    for (auto& b : fines_bytes) write_u64(out, b.size());
    write_u64(out, coarse_bytes.size());
    write_u64(out, tmpl_bytes.size());
    write_u64(out, meta.size());
    for (auto& b : fines_bytes) out.insert(out.end(), b.begin(), b.end());
    out.insert(out.end(), coarse_bytes.begin(), coarse_bytes.end());
    out.insert(out.end(), tmpl_bytes.begin(), tmpl_bytes.end());
    out.insert(out.end(), meta.begin(), meta.end());
    return out;
}

std::shared_ptr<TwoStageModel> decode_two_stage(const uint8_t* bytes, uint64_t size) {
    require(bytes && size <= 32ULL * 1024 * 1024, "invalid model buffer or size");
    if (size < kMagicLen || std::memcmp(bytes, kMagic, kMagicLen) != 0) return nullptr;
    size_t off = kMagicLen;
    if (off + kLenWidth > size) return nullptr;
    const uint64_t n = read_u64(bytes + off);
    off += kLenWidth;
    if (n == 0 || n > 64) return nullptr;
    if (off + (n + 3) * kLenWidth > size) return nullptr;
    std::vector<uint64_t> lens;
    for (uint64_t i = 0; i < n + 3; ++i) {
        lens.push_back(read_u64(bytes + off));
        off += kLenWidth;
    }
    std::vector<std::vector<uint8_t>> parts;
    for (uint64_t i = 0; i < n + 3; ++i) {
        if (lens[i] > size - off) return nullptr;
        parts.emplace_back(bytes + off, bytes + off + lens[i]);
        off += lens[i];
    }
    require(off == size, "container trailing bytes");
    auto model = std::make_shared<TwoStageModel>();
    for (uint64_t i = 0; i < n; ++i)
        model->fines.push_back(decode(parts[i].data(), parts[i].size()));
    model->coarse = decode(parts[n].data(), parts[n].size());
    require_png_template(parts[n + 1]);  // F09：先于 imdecode 的 PNG 格式+边长合同（内存峰值前置）
    model->tmpl_gray = cv::imdecode(parts[n + 1], cv::IMREAD_GRAYSCALE);
    if (model->tmpl_gray.empty()) return nullptr;
    const Meta meta = parse_meta(std::string(parts[n + 2].begin(), parts[n + 2].end()));
    model->k = meta.k;
    require(model->k == 2 || model->k == 4 || model->k == 8, "invalid coarse factor");
    model->groups = meta.groups;
    if (model->groups.size() != model->fines.size()) return nullptr;
    if (model->fines.empty() || !model->coarse) return nullptr;
    // F09：模板灰图资源封口（边≤4096、像素量 int64 限预算），先于 B7 的浮点图放大
    require(model->tmpl_gray.cols <= max_edge && model->tmpl_gray.rows <= max_edge,
            "template image exceeds edge budget");
    require(int64_t(model->tmpl_gray.cols) * model->tmpl_gray.rows <=
                int64_t(max_edge) * max_edge, "template image exceeds pixel budget");
    // F09：跨段一致性——各组 fine 必须同参考几何（防互不相关分段拼接）
    for (size_t i = 1; i < model->fines.size(); ++i)
        require(model->fines[i]->width == model->fines.front()->width &&
                    model->fines[i]->height == model->fines.front()->height,
                "inconsistent fine-group reference geometry");
    if (meta.has_cfg) {
        // F01：原始配置恢复 + F09 新容器强校验（ROI 对 tmpl、coarse 对 k 缩放关系）
        // R01 连带：cfg 三参是持久化实际值，先于 validate_config 严格校验——
        // 零/损坏须拒绝（与单级 codec 同口径），不得经训练零值默认语义洗白
        validate_persisted_detector_params(meta.cfg.feature_count, meta.cfg.weak_th,
                                           meta.cfg.strong_th);
        validate_config(meta.cfg, meta.ref_w, meta.ref_h);
        require(model->tmpl_gray.cols == meta.cfg.roi_width &&
                    model->tmpl_gray.rows == meta.cfg.roi_height,
                "template image disagrees with original training ROI");
        require(model->coarse->width == std::max(24, meta.ref_w / model->k) &&
                    model->coarse->height == std::max(24, meta.ref_h / model->k),
                "coarse segment disagrees with reference scale");
        // F09 跨段补核：fine 与 meta 的参考尺寸/ROI 原点/角度组逐项对齐（防
        // 各段互不相关的拼接容器）。角度容差 1e-9：段内 double 位级写读、
        // meta 17 位十进制往返均精确，1e-9 仅为防御性余量（实测差为 0）
        for (size_t i = 0; i < model->fines.size(); ++i) {
            const auto& fine = *model->fines[i];
            require(fine.width == meta.ref_w && fine.height == meta.ref_h,
                    "fine segment disagrees with reference size");
            require(fine.config.roi_x == meta.cfg.roi_x && fine.config.roi_y == meta.cfg.roi_y &&
                        fine.config.roi_width == meta.cfg.roi_width &&
                        fine.config.roi_height == meta.cfg.roi_height,
                    "fine segment disagrees with original training ROI");
            require(std::abs(fine.config.angle_min - model->groups[i].lo) <= 1e-9 &&
                        std::abs(fine.config.angle_max - model->groups[i].hi) <= 1e-9,
                    "fine segment angle range disagrees with group metadata");
            require(model->groups[i].lo >= meta.cfg.angle_min - 1e-9 &&
                        model->groups[i].hi <= meta.cfg.angle_max + 1e-9,
                    "group angle range outside original training range");
            // 三参一致性：fine 段解码回填的是 float 域实际生效值，cfg 为原样
            // double；在 float 域比较（两者均经同一 float 量化，非 float 级
            // 差异即为篡改/拼接）
            require(static_cast<float>(meta.cfg.weak_th) == fine.config.weak_th &&
                        static_cast<float>(meta.cfg.strong_th) == fine.config.strong_th &&
                        meta.cfg.feature_count == fine.config.feature_count,
                    "fine segment detector parameters disagree with original config");
        }
        model->base_config = meta.cfg;
        model->ref_width = meta.ref_w;
        model->ref_height = meta.ref_h;
    } else {
        // 旧容器无原始配置键：回退现行为——第一组 fine 的派生配置冒充原配置
        //（角度已细分、缩放已归一），ref=0 标记派生语义；旧资产仍可加载
        model->base_config = model->fines.front()->config;
    }
    model->b7 = extract_b7_points(model->tmpl_gray);
    return model;
}


namespace {

// 组选择（Python _fine_for_angle 对齐：重叠区取覆盖最多的一组）
const Model* fine_for_angle(const TwoStageModel& model, double angle, const GroupSpec** spec) {
    const Model* best = model.fines.front().get();
    const GroupSpec* best_spec = &model.groups.front();
    double best_cover = -1.0;
    for (size_t i = 0; i < model.fines.size(); ++i) {
        const auto& g = model.groups[i];
        if (g.lo - 1e-9 <= angle && angle <= g.hi + 1e-9) {
            const double cover = std::min(g.hi, angle + 180) - std::max(g.lo, angle - 180);
            if (cover > best_cover) {
                best_cover = cover;
                best = model.fines[i].get();
                best_spec = &g;
            }
        }
    }
    *spec = best_spec;
    return best;
}

void reproject_candidate(vf_shape_candidate& c, int x0, int y0,
                         const GroupSpec& g, double cs) {
    // Python _reproject 对齐（ds=1）：扣组偏置 → ×cs → +窗口原点；
    // transform 同链复合；scale=输出×cs。
    const double up = cs;
    c.center[0] = (c.center[0] - g.bias_x) * up + x0;
    c.center[1] = (c.center[1] - g.bias_y) * up + y0;
    for (int i = 0; i < 4; ++i) {
        c.corners[2 * i] = (c.corners[2 * i] - g.bias_x) * up + x0;
        c.corners[2 * i + 1] = (c.corners[2 * i + 1] - g.bias_y) * up + y0;
    }
    c.bbox[0] = (c.bbox[0] - g.bias_x) * up + x0;
    c.bbox[1] = (c.bbox[1] - g.bias_y) * up + y0;
    c.bbox[2] *= up;
    c.bbox[3] *= up;
    const double a = c.transform[0], b = c.transform[1], t_x = c.transform[2];
    const double d = c.transform[3], e = c.transform[4], t_y = c.transform[5];
    c.transform[0] = a * up;
    c.transform[1] = b * up;
    c.transform[2] = (t_x - g.bias_x) * up + x0;
    c.transform[3] = d * up;
    c.transform[4] = e * up;
    c.transform[5] = (t_y - g.bias_y) * up + y0;
    c.scale = c.scale * cs;
}

}  // namespace

namespace {
// 跨候选聚类（对齐单级路径 matcher.cpp 的同对象合并：重叠比 >0.4 合并，
// 姿态差异置歧义标志，簇内最高分为代表）——两级多窗口/跨组候选统一去重。
bool same_object(const vf_shape_candidate& a, const vf_shape_candidate& b) {
    const double iw =
        std::max(0., std::min(a.bbox[0] + a.bbox[2], b.bbox[0] + b.bbox[2]) -
                        std::max(a.bbox[0], b.bbox[0]));
    const double ih =
        std::max(0., std::min(a.bbox[1] + a.bbox[3], b.bbox[1] + b.bbox[3]) -
                        std::max(a.bbox[1], b.bbox[1]));
    const double overlap = iw * ih;
    const double area = a.bbox[2] * a.bbox[3] + b.bbox[2] * b.bbox[3] - overlap;
    return area > 0 && overlap / area > .4;
}

// 姿态歧义分数差豁免阈（0-100 相似度）：与 matcher.cpp 同值同语义，两处聚类
// 的豁免口径必须一致，改动需同步
constexpr double kAmbiguityScoreGap = 10.;

// 多目标搜索预算：粗候选按分数保留上限 / 精修窗调度上限。前两窗恒跑（保
// 双目标），其余窗串行补搜索并在达标去重候选数达 max_candidates 后早停
constexpr size_t kMaxCoarseCandidates = 16;
constexpr size_t kMaxFineWindows = 8;

std::vector<vf_shape_candidate> cluster_candidates(std::vector<vf_shape_candidate>&& raw) {
    std::vector<vf_shape_candidate> objects;
    for (auto& current : raw) {
        if (current.center[0] < 0 || current.center[1] < 0) continue;
        bool merged = false;
        for (auto& object : objects) {
            if (!same_object(object, current)) continue;
            const double angle =
                std::abs(std::remainder(object.angle - current.angle, 360.));
            // 语义对齐 matcher.cpp：姿态差超限且两解分数接近（差 < 豁免阈）才置
            // 歧义，分差拉开时高分解明显领先不置；标志只置位不清除（流式合并）
            if ((angle >= 15 || std::abs(std::log(object.scale / current.scale)) >= .14) &&
                std::abs(object.score - current.score) < kAmbiguityScoreGap)
                object.pose_ambiguous = 1;
            if (current.score > object.score) {
                current.pose_ambiguous = object.pose_ambiguous;
                object = current;
            }
            merged = true;
            break;
        }
        if (!merged) objects.push_back(current);
    }
    return objects;
}

// 早停计数与最终输出口径的单一实现（F08）：对候选集副本执行与收尾路径
// 完全相同的序列（分数降序排序→聚类→计过阈值数）。具名供见证测试
// 构造级对拍（tests/witness_test.cpp 直调）——勿在别处另行实现计数。
size_t count_qualified_objects(std::vector<vf_shape_candidate> candidates, double threshold) {
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    candidates = cluster_candidates(std::move(candidates));
    return size_t(std::count_if(candidates.begin(), candidates.end(),
                                [threshold](const vf_shape_candidate& c) {
                                    return c.score >= threshold;
                                }));
}
}  // namespace

// —— 两级匹配（Python TwoStageMatcher._match_locked 对齐）——
std::vector<vf_shape_candidate> match_two_stage(
    TwoStageModel& model, const cv::Mat& image_bgr, double threshold, int max_candidates) {
    require(std::isfinite(threshold) && threshold > 0 && threshold <= 100,
            "invalid similarity threshold");
    MatchProfile profile;
    const int w = image_bgr.cols, h = image_bgr.rows;
    // 粗层：1/k 缩图（4096 接口限制在编排内自然放开——缩图与窗口均小）
    cv::Mat small;
    MatchProfile::measure(profile.coarse_resize_ms, [&] {
      resize_coarse_image(image_bgr, small, model.k);
    });
    std::vector<vf_shape_candidate> coarse_cands;
    // 异常穿透到 ABI boundary 报错：执行失败不得伪装成"无匹配"空结果
    coarse_cands = MatchProfile::measure(profile.coarse_ms, [&] {
        return match(*model.coarse, small, std::max(45.0, threshold - 20.0));
    });
    if (coarse_cands.empty()) return {};
    std::sort(coarse_cands.begin(), coarse_cands.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    if (coarse_cands.size() > kMaxCoarseCandidates)
        coarse_cands.resize(kMaxCoarseCandidates);
    // 前两窗恒跑（Codex P1 教训：分差启发在双目标场景丢第二目标）；
    // 其余窗补足多目标覆盖，达标去重候选数达 max_candidates 早停
    const size_t use = std::min<size_t>(kMaxFineWindows, coarse_cands.size());
    profile.count = int(use);

    // 精修半窗自适应：模板对角/2 + 4k（窗口必须装得下模板）
    double fine_half = 300.0;
    if (!model.tmpl_gray.empty()) {
        const double diag = std::hypot(double(model.tmpl_gray.cols), double(model.tmpl_gray.rows));
        fine_half = std::max(240.0, diag / 2.0 + 4.0 * model.k);
    }
    const double fine_th = std::max(50.0, threshold * 0.85);
    auto refine_window = [&](size_t ci) {
        // 剖析槽仅 2 个（MatchProfile 契约）：前两窗入槽，其余窗记到本地空槽
        MatchProfile::Window overflow_trace;
        MatchProfile::Window& trace =
            ci < profile.windows.size() ? profile.windows[ci] : overflow_trace;
        std::vector<vf_shape_candidate> window_results;
        const auto& cc = coarse_cands[ci];
        const double cx = cc.center[0] * model.k, cy = cc.center[1] * model.k;
        double cs = (cc.scale > 0) ? cc.scale : 1.0;
        double ca = cc.angle;
        if (ca > 180) ca -= 360;
        const GroupSpec* spec = nullptr;
        const Model* fine_model = fine_for_angle(model, ca, &spec);

        const double half = fine_half * std::max(1.0, cs);
        const int x0 = int(std::max(0.0, cx - half)), y0 = int(std::max(0.0, cy - half));
        const int x1 = int(std::min(double(w), x0 + 2 * half));
        const int y1 = int(std::min(double(h), y0 + 2 * half));
        if (x1 - x0 < 24 || y1 - y0 < 24) return window_results;
        trace.width = x1 - x0; trace.height = y1 - y0;
        trace.scale = cs; trace.angle = ca;
        cv::Mat window = image_bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));

        cv::Mat window_scaled = window;
        const bool needs_scale = std::abs(cs - 1.0) > 0.02;
        if (needs_scale) {
            MatchProfile::Timer resize_timer(trace.resize_ms);
            cv::resize(window, window_scaled,
                       cv::Size(std::max(24, int(window.cols / cs)),
                                std::max(24, int(window.rows / cs))),
                       0, 0, cs > 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
        }
        cv::Mat effective = window_scaled;
        std::vector<vf_shape_candidate> refined;
        // 精修异常穿透（含并行窗 future.get 重抛）：失败窗口不得静默产出空结果
        auto best_score = [](const auto& candidates) {
            double score = 0;
            for (const auto& candidate : candidates) score = std::max(score, candidate.score);
            return score;
        };
        auto prepare = [&](const cv::Mat& image) {
            ++trace.prepares;
            return MatchProfile::measure(trace.prepare_ms, [&] {
                return prepare_frame(image, &fine_model->detector);
            });
        };
        auto evaluate = [&](Model& m, const PreparedFrame& frame) {
            const int ready = frame.features.preparedLevels();
            auto matches = MatchProfile::measure(trace.match_ms, [&] {
                return match_prepared(m, frame, fine_th);
            });
            trace.deferred_levels += frame.features.preparedLevels() - ready;
            return matches;
        };
        auto prepared = prepare(window_scaled);
        refined = evaluate(*const_cast<Model*>(fine_model), prepared);
        if (best_score(refined) < threshold && needs_scale) {
            ++trace.scale_retries;
            auto retry_frame = prepare(window);
            auto retry = evaluate(*const_cast<Model*>(fine_model), retry_frame);
            if (best_score(retry) > best_score(refined)) {
                cs = 1.0;
                effective = window;
                prepared = std::move(retry_frame);
                refined = std::move(retry);
            }
        }
        if (best_score(refined) < threshold && model.fines.size() > 1) {
            for (size_t gi = 0; gi < model.fines.size() && best_score(refined) < threshold; ++gi) {
                auto& alternate = *model.fines[gi];
                if (&alternate == fine_model) continue;
                ++trace.group_retries;
                const auto& cfg = alternate.config;
                const auto& original = fine_model->config;
                std::vector<vf_shape_candidate> retry;
                if (cfg.weak_th == original.weak_th && cfg.strong_th == original.strong_th &&
                    cfg.feature_count == original.feature_count) {
                    retry = evaluate(alternate, prepared);
                } else {
                    retry = match(alternate, effective, fine_th);
                }
                if (best_score(retry) > best_score(refined)) {
                    refined = std::move(retry);
                    spec = &model.groups[gi];
                }
            }
        }
        for (auto& r : refined) {
            reproject_candidate(r, x0, y0, *spec, cs);
            window_results.push_back(r);
        }
        return window_results;
    };
    std::vector<vf_shape_candidate> results;
    // F08：早停计数走 count_qualified_objects（与收尾路径同代码序列的具名
    // 单一实现）；IoU 非传递使任何单独实现的去重计数都与排序聚类不等价
    auto qualified_objects = [&results, threshold]() {
        return count_qualified_objects(results, threshold);
    };
    {
    MatchProfile::Timer window_timer(profile.windows_ms);
#ifdef VF_SHAPE_PARALLEL_WINDOWS
    std::future<std::vector<vf_shape_candidate>> second;
    if (use >= 2) second = std::async(std::launch::async, refine_window, 1);
    results = refine_window(0);
    if (second.valid()) {
        auto candidates = second.get();
        results.insert(results.end(), candidates.begin(), candidates.end());
    }
    for (size_t ci = 2; ci < use && qualified_objects() < size_t(max_candidates); ++ci) {
        auto candidates = refine_window(ci);
        results.insert(results.end(), candidates.begin(), candidates.end());
    }
#else
    for (size_t ci = 0; ci < use && (ci < 2 || qualified_objects() < size_t(max_candidates)); ++ci) {
        auto candidates = refine_window(ci);
        results.insert(results.end(), candidates.begin(), candidates.end());
    }
#endif
    }
    MatchProfile::Timer finish_timer(profile.finish_ms);
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    results = cluster_candidates(std::move(results));
    results.erase(std::remove_if(results.begin(), results.end(),
        [threshold](const auto& candidate) { return candidate.score < threshold; }), results.end());
    if (!results.empty() && !model.b7.px.empty()) {
        auto& best = results.front();
        double angle = best.angle > 180 ? best.angle - 360 : best.angle;
        const auto refined = b7_refine(image_bgr, model.b7, best.center[0], best.center[1],
                                       angle, best.scale);
        if (refined.valid >= kB7MinValidPts &&
            std::hypot(refined.x - best.center[0], refined.y - best.center[1]) <=
                2.0 * kB7SearchRadius) {
            apply_geometric_refinement(best, refined);
        }
    }
    if (results.size() > size_t(max_candidates)) results.resize(max_candidates);
    return results;
}

}  // namespace vfshape
