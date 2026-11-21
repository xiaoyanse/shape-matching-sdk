#pragma once

#include <array>
#include <atomic>
#ifdef VF_SHAPE_PROFILE_STAGES
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#endif

namespace vfshape {

// 匹配诊断快照（Plain 可复制载体，供 ABI 层落句柄档案）。
// 计数常开零开销；毫秒字段仅 VF_SHAPE_PROFILE_STAGES 构建填充，常规为 0。
struct MatchTraceSnapshot {
    int coarse_candidates = 0;
    int windows_searched = 0;
    int final_candidates = 0;
    std::array<double, 8> window_best{};
    // 收集阶段诊断：各窗重整前最高原始分（含未达判定阈者）与救活候选数
    std::array<double, 8> window_collect_best{};
    int rescued_candidates = 0;
    double coarse_ms = 0, coarse_resize_ms = 0, windows_ms = 0, finish_ms = 0;
    // 重试计数（前两窗剖析槽合计；溢出窗槽外不计，与 windows 数组契约同界）
    int scale_retries = 0, group_retries = 0;
};

class MatchProfile {
public:
    struct Window {
        double prepare_ms = 0, match_ms = 0, resize_ms = 0, scale = 0, angle = 0;
        int width = 0, height = 0, prepares = 0, scale_retries = 0, group_retries = 0;
        int deferred_levels = 0;
    };
    std::array<Window, 2> windows;
    // 常开计数：各精修窗最佳分（槽位上限 8，未执行窗为 0）；窗口 1 可能由
    // 并行线程写入，与窗口 0 分属不同元素、互不竞争
    std::array<double, 8> window_best{};
    // 各窗重整前最高原始分（写序同 window_best）；救活数为跨窗累加原子计数
    std::array<double, 8> window_collect_best{};
    std::atomic<int> rescued_candidates{0};
    std::atomic<int> windows_searched{0};
    int coarse_candidates = 0, final_candidates = 0;
    double coarse_ms = 0, coarse_resize_ms = 0, windows_ms = 0, finish_ms = 0;
    int count = 0;

    class Timer {
    public:
        explicit Timer(double& target)
#ifdef VF_SHAPE_PROFILE_STAGES
            : target_(target), start_(std::chrono::steady_clock::now())
#endif
        { (void)target; }
        ~Timer() {
#ifdef VF_SHAPE_PROFILE_STAGES
            target_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_).count();
#endif
        }
    private:
#ifdef VF_SHAPE_PROFILE_STAGES
        double& target_;
        std::chrono::steady_clock::time_point start_;
#endif
    };

    template<class Function>
    static auto measure(double& target, Function&& function) {
        Timer timer(target);
        return function();
    }

    MatchTraceSnapshot snapshot() const {
        MatchTraceSnapshot snap;
        snap.coarse_candidates = coarse_candidates;
        snap.windows_searched = windows_searched.load(std::memory_order_relaxed);
        snap.final_candidates = final_candidates;
        snap.window_best = window_best;
        snap.window_collect_best = window_collect_best;
        snap.rescued_candidates = rescued_candidates.load(std::memory_order_relaxed);
        snap.coarse_ms = coarse_ms;
        snap.coarse_resize_ms = coarse_resize_ms;
        snap.windows_ms = windows_ms;
        snap.finish_ms = finish_ms;
        snap.scale_retries = windows[0].scale_retries + windows[1].scale_retries;
        snap.group_retries = windows[0].group_retries + windows[1].group_retries;
        return snap;
    }

    ~MatchProfile() noexcept {
#ifdef VF_SHAPE_PROFILE_STAGES
        // 批量基准进程置 VF_SHAPE_TRACE_QUIET=1 免逐调用 JSON 淹没证据输出
        static const bool trace_quiet = std::getenv("VF_SHAPE_TRACE_QUIET") != nullptr;
        if (trace_quiet) return;
        try {
            std::ostringstream out;
            out << "VF_SHAPE_TRACE {\"coarse_ms\":" << coarse_ms + coarse_resize_ms
                << ",\"coarse_resize_ms\":" << coarse_resize_ms
                << ",\"windows_ms\":" << windows_ms << ",\"finish_ms\":" << finish_ms
                << ",\"count\":" << count << ",\"windows\":[";
            for (size_t i = 0; i < windows.size(); ++i) {
                const auto& w = windows[i];
                if (i) out << ',';
                out << "{\"width\":" << w.width << ",\"height\":" << w.height
                    << ",\"scale\":" << w.scale << ",\"angle\":" << w.angle
                    << ",\"prepares\":" << w.prepares << ",\"scale_retries\":" << w.scale_retries
                    << ",\"group_retries\":" << w.group_retries
                    << ",\"deferred_levels\":" << w.deferred_levels
                    << ",\"prepare_ms\":" << w.prepare_ms << ",\"match_ms\":" << w.match_ms
                    << ",\"resize_ms\":" << w.resize_ms << '}';
            }
            out << "]}\n";
            static std::mutex output_mutex;
            std::lock_guard<std::mutex> lock(output_mutex);
            std::fputs(out.str().c_str(), stdout);
            std::fflush(stdout);
        } catch (...) {}
#endif
    }
};
}
