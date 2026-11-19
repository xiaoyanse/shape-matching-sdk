#include "model_source.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpuspike {
namespace {

// 单级段签名（model_codec.cpp 常量，含结尾 NUL）
const char kSingleSignature[] = "VFSHAPE1:3df935f8e4b8c8a8ed72984211543e1b7696452e:p3";

class ByteReader {
public:
    ByteReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    bool u32(uint32_t& v) {
        if (size_ - pos_ < 4) return false;
        v = 0;
        for (int i = 0; i < 4; ++i) v |= uint32_t(data_[pos_++]) << (8 * i);
        return true;
    }
    bool u64(uint64_t& v) {
        if (size_ - pos_ < 8) return false;
        v = 0;
        for (int i = 0; i < 8; ++i) v |= uint64_t(data_[pos_++]) << (8 * i);
        return true;
    }
    bool f64(double& v) {
        uint64_t bits;
        if (!u64(bits)) return false;
        std::memcpy(&v, &bits, sizeof(v));
        return true;
    }
    size_t position() const { return pos_; }

private:
    const uint8_t* data_;
    size_t size_, pos_ = 0;
};

bool decode_single(const uint8_t* data, size_t size, std::vector<PoseModel>& poses,
                   double& weak, double& strong, int roi_out[4]) {
    const size_t sig_len = sizeof(kSingleSignature);  // 含 NUL
    if (size <= sig_len || std::memcmp(data, kSingleSignature, sig_len) != 0) return false;
    ByteReader r(data + sig_len, size - sig_len);
    uint32_t schema = 0;
    if (!r.u32(schema) || schema != 1) return false;
    uint32_t width = 0, height = 0;
    if (!r.u32(width) || !r.u32(height)) return false;
    for (int i = 0; i < 4; ++i) {  // roi
        uint32_t v;
        if (!r.u32(v)) return false;
        if (roi_out) roi_out[i] = int(v);
    }
    for (int i = 0; i < 6; ++i) {  // angle/scale 配置
        double v;
        if (!r.f64(v)) return false;
    }
    uint32_t features = 0;
    double weak_d = 0, strong_d = 0;
    if (!r.u32(features) || !r.f64(weak_d) || !r.f64(strong_d)) return false;
    weak = weak_d;
    strong = strong_d;
    uint32_t pose_count = 0;
    if (!r.u32(pose_count) || pose_count == 0 || pose_count > 350) return false;
    for (uint32_t p = 0; p < pose_count; ++p) {
        PoseModel pose;
        if (!r.f64(pose.angle) || !r.f64(pose.scale)) return false;
        uint32_t cw = 0, ch = 0;
        if (!r.u32(cw) || !r.u32(ch)) return false;
        for (int i = 0; i < 6; ++i) {  // transform
            double v;
            if (!r.f64(v)) return false;
            pose.transform[i] = v;
        }
        uint32_t levels = 0;
        if (!r.u32(levels) || levels != kPyramidLevels) return false;
        for (uint32_t l = 0; l < levels; ++l) {
            TemplLevel level;
            uint32_t w = 0, h = 0, tlx = 0, tly = 0, pl = 0, nf = 0;
            if (!r.u32(w) || !r.u32(h) || !r.u32(tlx) || !r.u32(tly) || !r.u32(pl)) return false;
            level.width = int(w);
            level.height = int(h);
            level.tl_x = int(tlx);
            level.tl_y = int(tly);
            if (!r.u32(nf)) return false;
            level.feats.resize(nf);
            for (uint32_t f = 0; f < nf; ++f) {
                uint32_t fx = 0, fy = 0, fl = 0;
                double theta = 0;
                if (!r.u32(fx) || !r.u32(fy) || !r.u32(fl) || !r.f64(theta)) return false;
                level.feats[f] = {int(fx), int(fy), int(fl & 7)};
            }
            pose.level[l] = std::move(level);
        }
        poses.push_back(std::move(pose));
    }
    return r.position() == size - sig_len;
}

std::vector<uint8_t> read_file(const char* path, std::string& err) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) {
        err = std::string("无法打开文件: ") + path;
        return {};
    }
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (size <= 0 || size > 64 * 1024 * 1024) {
        std::fclose(file);
        err = "文件尺寸异常";
        return {};
    }
    std::vector<uint8_t> bytes;
    bytes.resize(size_t(size));
    const size_t read = std::fread(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    if (read != bytes.size()) {
        err = "读取不完整";
        return {};
    }
    return bytes;
}

int scan_meta_k(const std::string& meta) {
    const size_t pos = meta.find("\"k\":");
    if (pos == std::string::npos) return 0;
    return std::atoi(meta.c_str() + pos + 4);
}
}  // namespace

bool load_model_set(const char* path, ModelSet& out, std::string& err) {
    const auto bytes = read_file(path, err);
    if (bytes.empty()) return false;
    if (bytes.size() < 13 || std::memcmp(bytes.data(), "VF2S3", 5) != 0) {
        err = "非 VF2S3 容器";
        return false;
    }
    ByteReader r(bytes.data() + 5, bytes.size() - 5);
    uint64_t n = 0;
    if (!r.u64(n) || n == 0 || n > 64) {
        err = "容器 fines 数量非法";
        return false;
    }
    std::vector<uint64_t> lens;
    for (uint64_t i = 0; i < n + 3; ++i) {
        uint64_t len = 0;
        if (!r.u64(len)) {
            err = "容器长度段截断";
            return false;
        }
        lens.push_back(len);
    }
    std::vector<std::pair<const uint8_t*, size_t>> parts;
    size_t offset = r.position() + 5;  // 相对容器头
    for (uint64_t i = 0; i < n + 3; ++i) {
        if (offset + lens[i] > bytes.size()) {
            err = "容器段越界";
            return false;
        }
        parts.emplace_back(bytes.data() + offset, size_t(lens[i]));
        offset += size_t(lens[i]);
    }
    if (offset != bytes.size()) {
        err = "容器尾部有残余字节";
        return false;
    }
    double weak = 30, strong = 60;
    for (uint64_t i = 0; i < n; ++i) {
        std::vector<PoseModel> poses;
        int roi[4] = {0, 0, 0, 0};
        if (!decode_single(parts[i].first, parts[i].second, poses, weak, strong, roi)) {
            err = "fine 段解码失败";
            return false;
        }
        if (i == 0) std::copy(roi, roi + 4, out.fine_roi);
        out.fines.push_back(std::move(poses));
    }
    if (!decode_single(parts[n].first, parts[n].second, out.coarse, weak, strong,
                       out.coarse_roi)) {
        err = "coarse 段解码失败";
        return false;
    }
    out.weak = weak;
    out.strong = strong;
    const std::string meta(reinterpret_cast<const char*>(parts[n + 2].first), parts[n + 2].second);
    const int k = scan_meta_k(meta);
    if (k == 2 || k == 4 || k == 8) out.k = k;
    for (const auto& group : out.fines) {
        if (group.empty()) {
            err = "空 fine 组";
            return false;
        }
        out.group_lo.push_back(group.front().angle);
        out.group_hi.push_back(group.back().angle);
    }
    return true;
}

bool load_frame(const char* path, int target_w, int target_h, FrameData& out, std::string& err) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        err = std::string("帧读取失败: ") + path;
        return false;
    }
    if (target_w > 0 && target_h > 0 && (image.cols != target_w || image.rows != target_h)) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
        image = resized;
    }
    cv::Mat continuous = image.isContinuous() ? image : image.clone();
    out.w = continuous.cols;
    out.h = continuous.rows;
    out.bgr.assign(continuous.data, continuous.data + continuous.total() * 3);
    return true;
}

bool shrink_frame(const FrameData& src, int factor, FrameData& out, std::string& err) {
    if (factor <= 0) {
        err = "缩放因子非法";
        return false;
    }
    const cv::Mat view(src.h, src.w, CV_8UC3, const_cast<uint8_t*>(src.bgr.data()));
    const int dw = std::max(24, src.w / factor), dh = std::max(24, src.h / factor);
    cv::Mat small;
    cv::resize(view, small, cv::Size(dw, dh), 0, 0, cv::INTER_AREA);
    out.w = small.cols;
    out.h = small.rows;
    out.bgr.assign(small.data, small.data + small.total() * 3);
    return true;
}

void pad_frame(const FrameData& src, FrameData& out) {
    constexpr int padding = 32;  // 引擎 PreparedFrame::padding
    const int w = (src.w + padding * 2 + 15) / 16 * 16;
    const int h = (src.h + padding * 2 + 15) / 16 * 16;
    out.w = w;
    out.h = h;
    out.bgr.assign(size_t(w) * h * 3, 0);
    for (int y = 0; y < src.h; ++y) {
        std::memcpy(out.bgr.data() + (size_t(y + padding) * w + padding) * 3,
                    src.bgr.data() + size_t(y) * src.w * 3, size_t(src.w) * 3);
    }
}

bool crop_window(const FrameData& src, double cx, double cy, double half, FrameData& out) {
    const int x0 = int(std::max(0.0, cx - half)), y0 = int(std::max(0.0, cy - half));
    const int x1 = int(std::min(double(src.w), x0 + 2 * half));
    const int y1 = int(std::min(double(src.h), y0 + 2 * half));
    if (x1 - x0 < 24 || y1 - y0 < 24) return false;
    out.w = x1 - x0;
    out.h = y1 - y0;
    out.bgr.resize(size_t(out.w) * out.h * 3);
    for (int y = 0; y < out.h; ++y)
        std::memcpy(out.bgr.data() + size_t(y) * out.w * 3,
                    src.bgr.data() + (size_t(y + y0) * src.w + x0) * 3, size_t(out.w) * 3);
    return true;
}

int pick_fine_group(const ModelSet& model, double angle) {
    int best = 0;
    double best_cover = -1.0;
    for (size_t i = 0; i < model.fines.size(); ++i) {
        const double lo = model.group_lo[i], hi = model.group_hi[i];
        if (lo - 1e-9 <= angle && angle <= hi + 1e-9) {
            const double cover = std::min(hi, angle + 180) - std::max(lo, angle - 180);
            if (cover > best_cover) {
                best_cover = cover;
                best = int(i);
            }
        }
    }
    return best;
}

}  // namespace gpuspike
