// 版本化容器（VF2S3）编解码与元数据解析。字节级兼容 Python coarse_to_fine；
// 从 two_stage.cpp 按职责拆出（编解码归本单元，训练与匹配留在 two_stage 系）。

#include "two_stage.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
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
    require_coded(png.size() >= 24, VF_SHAPE_MODEL_TAMPER,
                  "template image header too short for PNG", double(png.size()), 24.0);
    require_coded(std::memcmp(png.data(), signature, sizeof(signature)) == 0,
                  VF_SHAPE_MODEL_TAMPER, "template image header signature is not PNG");
    require_coded(std::memcmp(png.data() + 12, "IHDR", 4) == 0,
                  VF_SHAPE_MODEL_TAMPER, "template image header first chunk is not IHDR");
    auto u32_be = [&png](size_t off) {
        return (uint64_t(png[off]) << 24) | (uint64_t(png[off + 1]) << 16) |
               (uint64_t(png[off + 2]) << 8) | uint64_t(png[off + 3]);
    };
    require_coded(u32_be(16) <= uint64_t(max_edge) && u32_be(20) <= uint64_t(max_edge),
                  VF_SHAPE_MODEL_TAMPER, "template image header exceeds edge budget",
                  double(u32_be(16)), double(u32_be(20)), double(max_edge));
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
    require_coded(storage.isOpened(), VF_SHAPE_MODEL_FORMAT, "invalid container metadata");
    const auto factor = storage["k"];
    require_coded(factor.isInt(), VF_SHAPE_MODEL_FORMAT, "coarse factor must be an integer");
    Meta meta;
    meta.k = int(factor);
    require_coded(meta.k == 2 || meta.k == 4 || meta.k == 8, VF_SHAPE_MODEL_TAMPER,
                  "invalid coarse factor");
    const auto groups = storage["groups"], biases = storage["group_bias"];
    require_coded(groups.isSeq() && biases.isSeq() && groups.size() == biases.size() &&
                  groups.size() > 0 && groups.size() <= 64, VF_SHAPE_MODEL_FORMAT,
                  "invalid group metadata");
    auto number = [](const cv::FileNode& node) {
        require_coded(node.isInt() || node.isReal(), VF_SHAPE_MODEL_FORMAT,
                      "group metadata must be numeric");
        const double value = double(node);
        require_coded(std::isfinite(value), VF_SHAPE_MODEL_TAMPER,
                      "group metadata must be finite");
        return value;
    };
    auto integer = [&number](const cv::FileNode& node) {
        const double value = number(node);
        require_coded(value == std::floor(value) && std::abs(value) <= 2147483647.0,
                      VF_SHAPE_MODEL_TAMPER, "container metadata must be a 32-bit integer");
        return value;
    };
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto bounds = groups[int(i)], bias = biases[int(i)];
        require_coded(bounds.isSeq() && bounds.size() == 2 && bias.isSeq() && bias.size() == 2,
                      VF_SHAPE_MODEL_FORMAT, "invalid group bounds or anchor");
        GroupSpec group{number(bounds[0]), number(bounds[1]), number(bias[0]), number(bias[1])};
        require_coded(group.lo >= -180 && group.hi <= 180 && group.lo <= group.hi,
                      VF_SHAPE_MODEL_TAMPER, "invalid group angle range");
        require_coded(std::abs(group.bias_x) <= max_edge && std::abs(group.bias_y) <= max_edge,
                      VF_SHAPE_MODEL_TAMPER, "invalid group anchor");
        meta.groups.push_back(group);
    }
    // F01：可选原始配置键（旧容器缺失 → 派生回退；两键必须成对出现）
    const auto ref = storage["ref"], cfg = storage["cfg"];
    require_coded(ref.isNone() == cfg.isNone(), VF_SHAPE_MODEL_FORMAT,
                  "reference size and config must appear together");
    if (!ref.isNone()) {
        require_coded(ref.isSeq() && ref.size() == 2, VF_SHAPE_MODEL_FORMAT,
                      "invalid reference size metadata");
        const double rw = integer(ref[0]), rh = integer(ref[1]);
        require_coded(rw >= 24 && rw <= max_edge && rh >= 24 && rh <= max_edge,
                      VF_SHAPE_MODEL_TAMPER, "invalid reference size metadata");
        meta.ref_w = int(rw);
        meta.ref_h = int(rh);
        meta.has_ref = true;
    }
    if (!cfg.isNone()) {
        require_coded(cfg.isSeq() && cfg.size() == 13, VF_SHAPE_MODEL_FORMAT,
                      "invalid training config metadata");
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

}  // namespace

// —— 容器编解码 ——
std::vector<uint8_t> encode_two_stage(const TwoStageModel& model) {
    // 段：fines 导出 bytes ×n + coarse + tmpl_png + meta
    // F09：增量预算——每生成一段即累计，中途超 32MiB 立即抛（不等全部段生成
    // 完，缩短超限模型的编码期驻留）；合同与 decode 侧同一口径
    uint64_t total = kMagicLen + kLenWidth * (uint64_t(model.fines.size()) + 4);
    auto accrue = [&total](uint64_t bytes) {
        total += bytes;
        require_coded(total <= max_model_bytes, VF_SHAPE_MODEL_SIZE,
                      "model container exceeds size budget", double(total),
                      double(max_model_bytes));
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
    require_coded(bytes && size <= 32ULL * 1024 * 1024, VF_SHAPE_MODEL_SIZE,
                  "invalid model buffer or size", double(size), double(max_model_bytes));
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
    require_coded(off == size, VF_SHAPE_MODEL_TAMPER, "container trailing bytes");
    auto model = std::make_shared<TwoStageModel>();
    for (uint64_t i = 0; i < n; ++i)
        model->fines.push_back(decode(parts[i].data(), parts[i].size()));
    model->coarse = decode(parts[n].data(), parts[n].size());
    require_png_template(parts[n + 1]);  // F09：先于 imdecode 的 PNG 格式+边长合同（内存峰值前置）
    model->tmpl_gray = cv::imdecode(parts[n + 1], cv::IMREAD_GRAYSCALE);
    if (model->tmpl_gray.empty()) return nullptr;
    const Meta meta = parse_meta(std::string(parts[n + 2].begin(), parts[n + 2].end()));
    model->k = meta.k;
    require_coded(model->k == 2 || model->k == 4 || model->k == 8, VF_SHAPE_MODEL_TAMPER,
                  "invalid coarse factor");
    model->groups = meta.groups;
    if (model->groups.size() != model->fines.size()) return nullptr;
    if (model->fines.empty() || !model->coarse) return nullptr;
    // F09：模板灰图资源封口（边≤4096、像素量 int64 限预算），先于 B7 的浮点图放大
    require_coded(model->tmpl_gray.cols <= max_edge && model->tmpl_gray.rows <= max_edge,
                  VF_SHAPE_MODEL_TAMPER, "template image exceeds edge budget",
                  double(model->tmpl_gray.cols), double(model->tmpl_gray.rows),
                  double(max_edge));
    require_coded(int64_t(model->tmpl_gray.cols) * model->tmpl_gray.rows <=
                      int64_t(max_edge) * max_edge, VF_SHAPE_MODEL_TAMPER,
                  "template image exceeds pixel budget",
                  double(int64_t(model->tmpl_gray.cols) * model->tmpl_gray.rows),
                  double(int64_t(max_edge) * max_edge));
    // F09 跨段一致性——各组 fine 必须同参考几何（防互不相关分段拼接）
    for (size_t i = 1; i < model->fines.size(); ++i)
        require_coded(model->fines[i]->width == model->fines.front()->width &&
                      model->fines[i]->height == model->fines.front()->height,
                      VF_SHAPE_MODEL_TAMPER, "inconsistent fine-group reference geometry");
    if (meta.has_cfg) {
        // F01：原始配置恢复 + F09 新容器强校验（ROI 对 tmpl、coarse 对 k 缩放关系）
        // R01 连带：cfg 三参是持久化实际值，先于 validate_config 严格校验——
        // 零/损坏须拒绝（与单级 codec 同口径），不得经训练零值默认语义洗白
        validate_persisted_detector_params(meta.cfg.feature_count, meta.cfg.weak_th,
                                           meta.cfg.strong_th);
        validate_config(meta.cfg, meta.ref_w, meta.ref_h);
        require_coded(model->tmpl_gray.cols == meta.cfg.roi_width &&
                      model->tmpl_gray.rows == meta.cfg.roi_height,
                      VF_SHAPE_MODEL_TAMPER,
                      "template image disagrees with original training ROI");
        require_coded(model->coarse->width == std::max(24, meta.ref_w / model->k) &&
                      model->coarse->height == std::max(24, meta.ref_h / model->k),
                      VF_SHAPE_MODEL_TAMPER, "coarse segment disagrees with reference scale");
        // F09 跨段补核：fine 与 meta 的参考尺寸/ROI 原点/角度组逐项对齐（防
        // 各段互不相关的拼接容器）。角度容差 1e-9：段内 double 位级写读、
        // meta 17 位十进制往返均精确，1e-9 仅为防御性余量（实测差为 0）
        for (size_t i = 0; i < model->fines.size(); ++i) {
            const auto& fine = *model->fines[i];
            require_coded(fine.width == meta.ref_w && fine.height == meta.ref_h,
                          VF_SHAPE_MODEL_TAMPER, "fine segment disagrees with reference size");
            require_coded(fine.config.roi_x == meta.cfg.roi_x && fine.config.roi_y == meta.cfg.roi_y &&
                          fine.config.roi_width == meta.cfg.roi_width &&
                          fine.config.roi_height == meta.cfg.roi_height,
                          VF_SHAPE_MODEL_TAMPER,
                          "fine segment disagrees with original training ROI");
            require_coded(std::abs(fine.config.angle_min - model->groups[i].lo) <= 1e-9 &&
                          std::abs(fine.config.angle_max - model->groups[i].hi) <= 1e-9,
                          VF_SHAPE_MODEL_TAMPER,
                          "fine segment angle range disagrees with group metadata");
            require_coded(model->groups[i].lo >= meta.cfg.angle_min - 1e-9 &&
                          model->groups[i].hi <= meta.cfg.angle_max + 1e-9,
                          VF_SHAPE_MODEL_TAMPER,
                          "group angle range outside original training range");
            // 三参一致性：fine 段解码回填的是 float 域实际生效值，cfg 为原样
            // double；在 float 域比较（两者均经同一 float 量化，非 float 级
            // 差异即为篡改/拼接）
            require_coded(static_cast<float>(meta.cfg.weak_th) == fine.config.weak_th &&
                          static_cast<float>(meta.cfg.strong_th) == fine.config.strong_th &&
                          meta.cfg.feature_count == fine.config.feature_count,
                          VF_SHAPE_MODEL_TAMPER,
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

}  // namespace vfshape
