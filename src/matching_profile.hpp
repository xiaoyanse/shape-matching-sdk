#pragma once

#include <array>
#ifdef VF_SHAPE_PROFILE_STAGES
#include <chrono>
#include <cstdio>
#include <mutex>
#include <sstream>
#endif

namespace vfshape {
class MatchProfile {
public:
    struct Window {
        double prepare_ms = 0, match_ms = 0, resize_ms = 0, scale = 0, angle = 0;
        int width = 0, height = 0, prepares = 0, scale_retries = 0, group_retries = 0;
        int deferred_levels = 0;
    };
    std::array<Window, 2> windows;
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

    ~MatchProfile() noexcept {
#ifdef VF_SHAPE_PROFILE_STAGES
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
