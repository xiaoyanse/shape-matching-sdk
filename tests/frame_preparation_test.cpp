#include "../vendor/shape_based_matching/line2Dup.cpp"

#include <stdexcept>
#include <future>

namespace {
void equal(const cv::Mat& actual, const cv::Mat& expected) {
    if (actual.size() != expected.size() || actual.type() != expected.type() ||
        cv::norm(actual, expected, cv::NORM_INF) != 0)
        throw std::runtime_error("frame preparation changed a response byte");
}

cv::Mat reference_spread(const cv::Mat& input, int step) {
    cv::Mat output = cv::Mat::zeros(input.size(), CV_8U);
    for (int y = 0; y < input.rows; ++y)
        for (int x = 0; x < input.cols; ++x)
            for (int dy = 0; dy < step && y + dy < input.rows; ++dy)
                for (int dx = 0; dx < step && x + dx < input.cols; ++dx)
                    output.at<uchar>(y, x) |= input.at<uchar>(y + dy, x + dx);
    return output;
}

void responses(const cv::Mat& input, int step) {
    std::vector<cv::Mat> maps, actual;
    line2Dup::computeResponseMaps(input, maps);
    line2Dup::linearizeResponses(input, actual, step);
    for (int ori = 0; ori < 8; ++ori) {
        cv::Mat expected;
        line2Dup::linearize(maps[ori], expected, step);
        equal(actual[ori], expected);
    }
}

void histogram(cv::RNG& random) {
    cv::Mat angles(113, 157, CV_32F), magnitudes(113, 157, CV_32F);
    random.fill(angles, cv::RNG::UNIFORM, 0.f, 360.f);
    random.fill(magnitudes, cv::RNG::UNIFORM, 0.f, 2000.f);
    for (int label = 0; label < 8; ++label)
        angles(cv::Rect(label * 18, 55, 18, 58)).setTo(label * 22.5f);
    magnitudes.row(60).setTo(900.f);
    cv::Mat bins;
    angles.convertTo(bins, CV_8U, 16.0 / 360.0);
    bins.row(0).setTo(0); bins.row(bins.rows - 1).setTo(0);
    bins.col(0).setTo(0); bins.col(bins.cols - 1).setTo(0);
    for (int y = 1; y < bins.rows - 1; ++y)
        for (int x = 1; x < bins.cols - 1; ++x) bins.at<uchar>(y, x) &= 7;
    for (float threshold : {0.f, 900.f, 2000.f}) {
        cv::Mat expected = cv::Mat::zeros(bins.size(), CV_8U), actual;
        for (int y = 1; y < bins.rows - 1; ++y)
            for (int x = 1; x < bins.cols - 1; ++x) {
                if (!(magnitudes.at<float>(y, x) > threshold)) continue;
                std::array<int, 8> votes{};
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        ++votes[bins.at<uchar>(y + dy, x + dx)];
                const auto best = std::max_element(votes.begin(), votes.end());
                if (*best >= 5) expected.at<uchar>(y, x) = uchar(1 << (best - votes.begin()));
            }
        line2Dup::hysteresisGradient(magnitudes, actual, angles, threshold);
        equal(actual, expected);
    }
}

void gradients(cv::RNG& random) {
    for (const cv::Size size : {cv::Size(1, 1), cv::Size(2, 3), cv::Size(8, 7),
             cv::Size(9, 17), cv::Size(10, 3), cv::Size(16, 9), cv::Size(17, 9),
             cv::Size(24, 3), cv::Size(25, 4), cv::Size(63, 81)}) {
        cv::Mat storage(size.height + 6, size.width + 4, CV_8UC3);
        random.fill(storage, cv::RNG::UNIFORM, 0, 256);
        cv::Mat image = storage(cv::Rect(2, 3, size.width, size.height));
        cv::Mat reference_x, reference_y, gx, gy, magnitude;
        cv::Sobel(image.clone(), reference_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::Sobel(image.clone(), reference_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
        line2Dup::strongestGradient(image, magnitude, gx, gy);
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x) {
                const auto dx = reference_x.at<cv::Vec3s>(y, x);
                const auto dy = reference_y.at<cv::Vec3s>(y, x);
                int best = 0;
                for (int c = 1; c < 3; ++c)
                    if (dx[c] * dx[c] + dy[c] * dy[c] > dx[best] * dx[best] + dy[best] * dy[best]) best = c;
                if (gx.at<float>(y, x) != dx[best] || gy.at<float>(y, x) != dy[best] ||
                    magnitude.at<float>(y, x) != dx[best] * dx[best] + dy[best] * dy[best])
                    throw std::runtime_error("fused Sobel differs from OpenCV");
            }
    }
}

void deferred_preparation(cv::RNG& random) {
    cv::Mat source(256, 256, CV_8UC3), mask = cv::Mat::zeros(256, 256, CV_8U);
    random.fill(source, cv::RNG::UNIFORM, 0, 256);
    mask(cv::Rect(48, 48, 128, 128)).setTo(255);
    line2Dup::Detector detector(128, {4, 8}, 30, 60);
    if (detector.addTemplate(source, "shape", mask) < 0)
        throw std::runtime_error("deferred preparation fixture has no template");
    auto eager = detector.prepare(source);
    auto deferred = detector.prepare(source, {}, false);
    auto shared = deferred;
    if (eager.preparedLevels() != 2 || deferred.preparedLevels() != 1)
        throw std::runtime_error("deferred preparation did not defer the fine level");
    const auto expected = detector.matchPrepared(eager, 80);
    if (expected.empty()) throw std::runtime_error("eager preparation lost fixture");
    cv::Mat owned = source.clone();
    auto owned_frame = detector.prepareOwned(std::move(owned));
    if (!owned.empty() || detector.matchPrepared(owned_frame, 80) != expected)
        throw std::runtime_error("owned preparation did not transfer private pixels correctly");
    cv::Mat aliased_source = source.clone(), alias = aliased_source;
    auto aliased_frame = detector.prepareOwned(std::move(aliased_source));
    alias.setTo(0);
    if (!aliased_source.empty() || detector.matchPrepared(aliased_frame, 80) != expected)
        throw std::runtime_error("shared owned input did not retain snapshot protection");
    cv::Mat external_storage = source.clone();
    cv::Mat borrowed(source.rows, source.cols, source.type(), external_storage.data, external_storage.step);
    auto borrowed_frame = detector.prepareOwned(std::move(borrowed));
    external_storage.setTo(0);
    if (detector.matchPrepared(borrowed_frame, 80) != expected)
        throw std::runtime_error("non-owning input did not retain snapshot protection");
    auto masked_eager = detector.prepare(source, mask);
    auto masked_deferred = detector.prepare(source, mask, false);
    const auto expected_masked = detector.matchPrepared(masked_eager, 80);
    if (expected_masked.empty()) throw std::runtime_error("masked fixture has no matches");
    mask.setTo(0);
    if (detector.matchPrepared(masked_deferred, 80) != expected_masked)
        throw std::runtime_error("deferred mask snapshot changed matches");
    source.setTo(0);
    auto concurrent = std::async(std::launch::async, [&] {
        return detector.matchPrepared(shared, 80);
    });
    if (detector.matchPrepared(deferred, 80) != expected || concurrent.get() != expected ||
        deferred.preparedLevels() != 2 || shared.preparedLevels() != 2)
        throw std::runtime_error("deferred copy, snapshot or concurrent materialization changed matches");
    auto empty = detector.prepare(source, {}, false);
    if (!detector.matchPrepared(empty, 80).empty() || empty.preparedLevels() != 1)
        throw std::runtime_error("empty frame unnecessarily materialized fine responses");
    cv::Mat parent(288, 288, CV_8UC3);
    random.fill(parent, cv::RNG::UNIFORM, 0, 256);
    if (detector.prepare(parent(cv::Rect(16, 16, 256, 256)), {}, false).preparedLevels() != 2)
        throw std::runtime_error("ROI preparation did not preserve parent border semantics");
}

void regional_preparation(cv::RNG& random) {
    for (int channels : {1, 3}) {
        cv::Mat source(128, 160, CV_MAKETYPE(CV_8U, channels)), mask(128, 160, CV_8U);
        random.fill(source, cv::RNG::UNIFORM, 0, 256);
        random.fill(mask, cv::RNG::UNIFORM, 0, 2);
        for (bool masked : {false, true})
            for (int step : {4, 8}) {
                const cv::Mat effective_mask = masked ? mask : cv::Mat();
                std::vector<cv::Mat> expected;
                line2Dup::prepareMemories(source, effective_mask, step, 30, 128, 60, expected);
                for (const cv::Rect region : {cv::Rect(0, 0, 1, 1), cv::Rect(159, 127, 1, 1),
                         cv::Rect(7, 13, 33, 46), cv::Rect(20, 20, 120, 88), cv::Rect(0, 0, 160, 128)}) {
                    std::vector<cv::Mat> actual(8);
                    for (auto& memory : actual)
                        memory = cv::Mat(step * step, (160 / step) * (128 / step), CV_8U, cv::Scalar(170));
                    line2Dup::prepareMemoryRegion(source, effective_mask, step, 30, 128, 60, region, actual);
                    for (int ori = 0; ori < 8; ++ori)
                        for (int y = 0; y < 128; ++y)
                            for (int x = 0; x < 160; ++x) {
                                const int row = y % step * step + x % step;
                                const int col = y / step * (160 / step) + x / step;
                                const uchar value = region.contains(cv::Point(x, y)) ? expected[ori].at<uchar>(row, col) : 170;
                                if (actual[ori].at<uchar>(row, col) != value)
                                    throw std::runtime_error("regional preparation changed a response or wrote outside request");
                            }
                }
            }
    }
}
}

int main() {
    try {
        cv::RNG random(56789);
        histogram(random);
        gradients(random);
        deferred_preparation(random);
        regional_preparation(random);
        int checked = 0;
        for (int step : {1, 2, 4, 8}) {
            for (const cv::Size size : {cv::Size(16, 16), cv::Size(80, 48), cv::Size(128, 96)}) {
                cv::Mat storage(size.height + 6, size.width + 9, CV_8U);
                random.fill(storage, cv::RNG::UNIFORM, 0, 256);
                cv::Mat input = storage(cv::Rect(3, 2, size.width, size.height));
                cv::Mat actual;
                line2Dup::spread(input, actual, step);
                equal(actual, reference_spread(input, step));
                responses(actual, step);
                ++checked;
            }
        }
        cv::Mat all_values(16, 16, CV_8U);
        for (int i = 0; i < 256; ++i) all_values.ptr()[i] = uchar(i);
        for (int step : {1, 2, 4, 8}) responses(all_values, step);
        for (int value : {0, 1, 128, 255}) {
            cv::Mat input(32, 32, CV_8U, cv::Scalar(value)), actual;
            line2Dup::spread(input, actual, 8);
            equal(actual, reference_spread(input, 8));
        }
        std::cout << "frame response contracts: " << checked << " strided cases, all LUT values, borders, histogram\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
