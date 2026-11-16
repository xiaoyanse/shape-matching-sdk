// 早停计数口径见证（构造级）：包含 src/two_stage.cpp 编译，直接取匿名命名
// 空间的 same_object / cluster_candidates，在复审分歧见证序列上断言
// "排序→聚类→计数"路径与最终输出口径给出同一对象数（旧的插入序贪心口径
// 在同数据上给出不同数，作为回退哨兵一并钉死）。
#include "../src/two_stage.cpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

vf_shape_candidate box(double x, double score) {
    vf_shape_candidate c{};
    c.bbox[0] = x;
    c.bbox[1] = 100;
    c.bbox[2] = 100;
    c.bbox[3] = 100;
    c.center[0] = x + 50;
    c.center[1] = 150;
    c.score = score;
    c.angle = 0;
    c.scale = 1;
    return c;
}

// 与 match_two_stage 早停闭包/收尾路径逐字一致的口径
size_t early_stop_count(std::vector<vf_shape_candidate> sequence, double threshold) {
    std::sort(sequence.begin(), sequence.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    auto clustered = vfshape::cluster_candidates(std::move(sequence));
    return size_t(std::count_if(clustered.begin(), clustered.end(),
                                [threshold](const vf_shape_candidate& c) {
                                    return c.score >= threshold;
                                }));
}

size_t final_count(std::vector<vf_shape_candidate> sequence, double threshold) {
    std::sort(sequence.begin(), sequence.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    auto clustered = vfshape::cluster_candidates(std::move(sequence));
    clustered.erase(std::remove_if(clustered.begin(), clustered.end(),
                                   [threshold](const auto& candidate) {
                                       return candidate.score < threshold;
                                   }),
                    clustered.end());
    return clustered.size();
}

size_t legacy_insertion_count(const std::vector<vf_shape_candidate>& sequence,
                              double threshold) {
    std::vector<const vf_shape_candidate*> kept;
    for (const auto& c : sequence) {
        if (c.score < threshold) continue;
        bool duplicate = false;
        for (const auto* k : kept)
            if (vfshape::same_object(*k, c)) {
                duplicate = true;
                break;
            }
        if (!duplicate) kept.push_back(&c);
    }
    return kept.size();
}

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        const double threshold = 60.0;
        // 见证一（复审分歧序列）：插入序 A,B,C,D——A(100,70)/B(160,80) 重叠
        // 0.25 各自保留、C(130,90) 被插入序吞；排序口径 A 被簇代表 C 吸收。
        const std::vector<vf_shape_candidate> witness = {box(100, 70), box(160, 80),
                                                         box(130, 90), box(500, 95)};
        const size_t legacy = legacy_insertion_count(witness, threshold);
        const size_t counted = early_stop_count(witness, threshold);
        const size_t final = final_count(witness, threshold);
        check(legacy == 3, "旧插入序口径见证值漂移：" + std::to_string(legacy) + " 应为 3");
        check(counted == 2 && final == 2,
              "三框分歧见证计数漂移：早停 " + std::to_string(counted) + " 最终 " +
                  std::to_string(final) + " 应均为 2");
        check(counted == final, "早停口径与最终口径不一致");

        // 见证二（分离多框）：无簇合并时两口径同数（路径一致冒烟）
        const std::vector<vf_shape_candidate> separated = {box(100, 95), box(400, 80),
                                                           box(700, 70)};
        check(early_stop_count(separated, threshold) == 3 &&
                  final_count(separated, threshold) == 3 &&
                  legacy_insertion_count(separated, threshold) == 3,
              "分离三框计数应为 3/3/3");

        std::cout << "witness: PASS (legacy=" << legacy << " counted=" << counted
                  << " final=" << final << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
