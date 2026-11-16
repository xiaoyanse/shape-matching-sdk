#include "model.hpp"
#include <cmath>
#include <cstring>
#include <limits>

namespace vfshape {
namespace {
constexpr char signature[] = "VFSHAPE1:3df935f8e4b8c8a8ed72984211543e1b7696452e:p3";
class Writer {
public:
    std::vector<uint8_t> bytes;
    void integer(uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) bytes.push_back(uint8_t(value >> shift));
    }
    void number(double value) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        integer(uint32_t(bits)); integer(uint32_t(bits >> 32));
    }
};
class Reader {
    const uint8_t* bytes_;
    uint64_t size_, position_ = 0;
public:
    Reader(const uint8_t* bytes, uint64_t size) : bytes_(bytes), size_(size) {}
    uint32_t integer() {
        require(size_ - position_ >= 4, "truncated model");
        uint32_t value = 0;
        for (int shift = 0; shift < 32; shift += 8) value |= uint32_t(bytes_[position_++]) << shift;
        return value;
    }
    int bounded(int low, int high) {
        const auto value = integer();
        require(value >= uint32_t(low) && value <= uint32_t(high), "model integer outside bounds");
        return int(value);
    }
    double number() {
        uint64_t bits = integer(); bits |= uint64_t(integer()) << 32;
        double value; std::memcpy(&value, &bits, sizeof(value));
        require(std::isfinite(value), "non-finite model number");
        return value;
    }
    bool finished() const { return position_ == size_; }
};
void write_config(Writer& out, const vf_shape_train_config& c) {
    out.integer(c.roi_x); out.integer(c.roi_y);
    out.integer(c.roi_width); out.integer(c.roi_height);
    for (auto value : {c.angle_min, c.angle_max, c.angle_step,
                       c.scale_min, c.scale_max, c.scale_step}) out.number(value);
}
vf_shape_train_config read_config(Reader& in) {
    vf_shape_train_config c{}; c.struct_size = sizeof(c);
    c.roi_x = in.bounded(0, max_edge); c.roi_y = in.bounded(0, max_edge);
    c.roi_width = in.bounded(24, max_edge); c.roi_height = in.bounded(24, max_edge);
    c.angle_min = in.number(); c.angle_max = in.number(); c.angle_step = in.number();
    c.scale_min = in.number(); c.scale_max = in.number(); c.scale_step = in.number();
    return c;
}
int grid_index(const std::vector<double>& values, double value) {
    for (size_t index = 0; index < values.size(); ++index)
        if (std::abs(values[index] - value) < 1e-7) return int(index);
    return -1;
}
}

std::vector<uint8_t> encode(const Model& model) {
    Writer out;
    out.integer(1); out.integer(model.width); out.integer(model.height);
    write_config(out, model.config);
    // detector 参数随模型序列化（自包含；decode 侧按值重建）
    const int enc_features = model.config.feature_count > 0 ? model.config.feature_count : feature_count;
    const float enc_weak = model.config.weak_th > 0 ? float(model.config.weak_th) : weak_threshold;
    const float enc_strong = model.config.strong_th > 0 ? float(model.config.strong_th) : strong_threshold;
    out.integer(enc_features); out.number(enc_weak); out.number(enc_strong);
    out.integer(uint32_t(model.poses.size()));
    for (size_t id = 0; id < model.poses.size(); ++id) {
        const auto& pose = model.poses[id];
        out.number(pose.angle); out.number(pose.scale);
        out.integer(pose.canvas_width); out.integer(pose.canvas_height);
        for (double value : pose.transform) out.number(value);
        const auto& pyramid = model.detector.getTemplates("shape", int(id));
        out.integer(uint32_t(pyramid.size()));
        for (const auto& item : pyramid) {
            out.integer(item.width); out.integer(item.height);
            out.integer(item.tl_x); out.integer(item.tl_y); out.integer(item.pyramid_level);
            out.integer(uint32_t(item.features.size()));
            for (const auto& f : item.features) {
                out.integer(f.x); out.integer(f.y); out.integer(f.label); out.number(f.theta);
            }
        }
    }
    out.bytes.insert(out.bytes.begin(), signature, signature + sizeof(signature));
    require(out.bytes.size() <= max_model_bytes, "model byte budget exceeded");
    return std::move(out.bytes);
}

std::shared_ptr<Model> decode(const uint8_t* bytes, uint64_t size) {
    require(bytes && size > sizeof(signature) && size <= max_model_bytes, "invalid model size");
    require(std::memcmp(bytes, signature, sizeof(signature)) == 0, "unsupported model source/format");
    Reader in(bytes + sizeof(signature), size - sizeof(signature));
    require(in.integer() == 1, "unsupported model schema");
    const int dec_width = in.bounded(24, max_edge), dec_height = in.bounded(24, max_edge);
    auto config = read_config(in);
    validate_config(config, dec_width, dec_height);
    const int dec_features = in.bounded(16, max_features_per_level);
    const double dec_weak = in.number(), dec_strong = in.number();
    require(0 < dec_weak && dec_weak <= dec_strong && dec_strong <= 255,
            "unsupported detector parameters");
    // 参数化重建（旧模型三值=训练期常量，读回即原值）
    auto model = std::make_shared<Model>(dec_features, float(dec_weak), float(dec_strong));
    model->width = dec_width; model->height = dec_height; model->config = config;
    const auto angles = samples(model->config.angle_min, model->config.angle_max,
                               model->config.angle_step, -180, 180);
    const auto scales = samples(model->config.scale_min, model->config.scale_max,
                               model->config.scale_step, .25, 2);
    const int count = in.bounded(1, max_poses);
    require(size_t(count) <= angles.size() * scales.size(), "pose count exceeds training grid");
    require(size >= uint64_t(count) * 140, "truncated pose collection");
    std::vector<std::pair<int, int>> seen;
    for (int id = 0; id < count; ++id) {
        Pose pose;
        pose.angle = in.number(); pose.scale = in.number();
        const int ai = grid_index(angles, pose.angle), si = grid_index(scales, pose.scale);
        require(ai >= 0 && si >= 0, "pose outside training grid");
        for (const auto& old : seen) require(old != std::make_pair(ai, si), "duplicate pose grid index");
        seen.emplace_back(ai, si);
        pose.canvas_width = in.bounded(24, max_edge); pose.canvas_height = in.bounded(24, max_edge);
        for (auto& value : pose.transform) value = in.number();
        const auto& c = model->config;
        const int sw = cvRound(c.roi_width * pose.scale), sh = cvRound(c.roi_height * pose.scale);
        const int padding = int(std::ceil(std::max(sw, sh) * std::sqrt(2.) / 2.)) + 24;
        require(pose.canvas_width == sw + padding * 2 && pose.canvas_height == sh + padding * 2,
                "invalid pose canvas");
        const auto rotation = cv::getRotationMatrix2D(
            cv::Point2f(pose.canvas_width / 2.f, pose.canvas_height / 2.f), pose.angle, 1);
        for (int row = 0; row < 2; ++row) {
            const double a = rotation.at<double>(row, 0), b = rotation.at<double>(row, 1);
            const double expected[] = {a * sw / c.roi_width, b * sh / c.roi_height,
                rotation.at<double>(row, 2) + a * (padding - double(sw) * c.roi_x / c.roi_width)
                + b * (padding - double(sh) * c.roi_y / c.roi_height)};
            for (int col = 0; col < 3; ++col)
                require(std::abs(pose.transform[row * 3 + col] - expected[col]) < 1e-6,
                        "pose transform disagrees with reference geometry");
        }
        require(in.integer() == 2, "invalid pyramid depth");
        std::vector<line2Dup::Template> pyramid;
        for (int level = 0; level < 2; ++level) {
            line2Dup::Template item{};
            const int cw = pose.canvas_width >> level, ch = pose.canvas_height >> level;
            item.width = in.bounded(1, cw); item.height = in.bounded(1, ch);
            item.tl_x = in.bounded(0, cw); item.tl_y = in.bounded(0, ch);
            require(item.width + item.tl_x <= cw && item.height + item.tl_y <= ch,
                    "template rectangle outside canvas");
            item.pyramid_level = in.bounded(level, level);
            if (level) {
                const auto& base = pyramid.front();
                require(item.width == (base.width >> level) && item.height == (base.height >> level)
                        && item.tl_x == (base.tl_x >> level) && item.tl_y == (base.tl_y >> level),
                        "inconsistent pyramid geometry");
            }
            const int features = in.bounded(4, max_features_per_level);
            for (int n = 0; n < features; ++n) {
                line2Dup::Feature f;
                f.x = in.bounded(0, item.width); f.y = in.bounded(0, item.height);
                f.label = in.bounded(0, 7); f.theta = float(in.number());
                require(f.theta >= 0 && f.theta <= 360, "invalid feature orientation");
                item.features.push_back(f);
            }
            pyramid.push_back(std::move(item));
        }
        model->poses.push_back(pose); model->detector.append(std::move(pyramid));
    }
    require(in.finished(), "model trailing bytes");
    // 角度插值未消费，解码侧不回填 adjacency（邻居计算跳过）。
    // model->build_pose_adjacency();
    return model;
}
}
