#include "line2Dup.h"
#include <iostream>
#include <array>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

using namespace std;
using namespace cv;

#include <chrono>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
#ifdef VF_SHAPE_PROFILE_STDOUT
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
#endif
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace line2Dup
{
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << "]";
}

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i)
    {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;

    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i)
    {
        features[i].write(fs);
    }
    fs << "]"; // features
}

static Rect cropTemplates(std::vector<Template> &templates)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y  >> templ.pyramid_level;

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            templ.features[j].x -= templ.tl_x;
            templ.features[j].y -= templ.tl_y;
        }
    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                   std::vector<Feature> &features,
                                                   size_t num_features, float distance)
{
    features.clear();
    float distance_sq = distance * distance;
    int i = 0;

    bool first_select = true;

    while(true)
    {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j)
        {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep)
            features.push_back(c.f);

        if (++i == (int)candidates.size()){
            bool num_ok = features.size() >= num_features;

            if(first_select){
                if(num_ok){
                    features.clear(); // we don't want too many first time
                    i = 0;
                    distance += 1.0f;
                    distance_sq = distance * distance;
                    continue;
                }else{
                    first_select = false;
                }
            }

            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
             if (num_ok || distance < 3){
                 break;
             }
        }
    }
    return true;
}

/****************************************************************************************\
*                                                         Color gradient ColorGradient                                                                        *
\****************************************************************************************/

void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
                        Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
            quant_r[c] &= 7;
    }

    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    parallel_for_(Range(1, angle.rows - 1), [&](const Range& rows) {
      for (int r = rows.start; r < rows.end; ++r)
      {
        const float *mag_r = magnitude.ptr<float>(r);
        int c = 1;
#if defined(__SSE2__) || defined(_M_X64)
        for (; c + 16 < angle.cols; c += 16) {
            __m128i neighbors[9];
            int k = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    neighbors[k++] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        quantized_unfiltered.ptr(r + dy) + c + dx));
            __m128i labels = _mm_setzero_si128();
            for (int orientation = 0; orientation < 8; ++orientation) {
                __m128i count = _mm_setzero_si128();
                const __m128i bin = _mm_set1_epi8(char(orientation));
                for (const auto neighbor : neighbors)
                    count = _mm_sub_epi8(count, _mm_cmpeq_epi8(neighbor, bin));
                // Nine votes admit at most one bin with five or more votes.
                const __m128i majority = _mm_cmpgt_epi8(count, _mm_set1_epi8(4));
                labels = _mm_or_si128(labels,
                    _mm_and_si128(majority, _mm_set1_epi8(char(1 << orientation))));
            }
            const __m128 limit = _mm_set1_ps(threshold);
            const __m128i lo = _mm_packs_epi32(
                _mm_castps_si128(_mm_cmpgt_ps(_mm_loadu_ps(mag_r + c), limit)),
                _mm_castps_si128(_mm_cmpgt_ps(_mm_loadu_ps(mag_r + c + 4), limit)));
            const __m128i hi = _mm_packs_epi32(
                _mm_castps_si128(_mm_cmpgt_ps(_mm_loadu_ps(mag_r + c + 8), limit)),
                _mm_castps_si128(_mm_cmpgt_ps(_mm_loadu_ps(mag_r + c + 12), limit)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(quantized_angle.ptr(r) + c),
                _mm_and_si128(labels, _mm_packs_epi16(lo, hi)));
        }
#endif
        for (; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 8; ++i)
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = 5;
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
      }
    });
}

static void strongestGradient(const Mat& image, Mat& magnitude, Mat& dx, Mat& dy)
{
    magnitude.create(image.size(), CV_32F);
    dx.create(image.size(), CV_32F);
    dy.create(image.size(), CV_32F);
    parallel_for_(Range(0, image.rows), [&](const Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            const uchar* top = image.ptr(std::max(0, y - 1));
            const uchar* mid = image.ptr(y);
            const uchar* bottom = image.ptr(std::min(image.rows - 1, y + 1));
            float* out_x = dx.ptr<float>(y);
            float* out_y = dy.ptr<float>(y);
            float* out_mag = magnitude.ptr<float>(y);
            for (int x = 0; x < image.cols; ++x) {
#if defined(__SSE2__) || defined(_M_X64)
                if (x > 0 && x + 8 < image.cols) {
                    alignas(16) short gradient_x[24], gradient_y[24];
                    alignas(16) int norms[24];
                    const __m128i zero = _mm_setzero_si128();
                    auto load = [&](const uchar* row, int offset) {
                        return _mm_unpacklo_epi8(_mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(row + offset)), zero);
                    };
                    for (int c = 0; c < 24; c += 8) {
                        const int p = 3 * x + c;
                        const auto tl = load(top, p - 3), tr = load(top, p + 3);
                        const auto bl = load(bottom, p - 3), br = load(bottom, p + 3);
                        const auto middle_dx = _mm_sub_epi16(load(mid, p + 3), load(mid, p - 3));
                        const auto middle_dy = _mm_sub_epi16(load(bottom, p), load(top, p));
                        const auto gx = _mm_add_epi16(_mm_add_epi16(_mm_sub_epi16(tr, tl),
                            _mm_slli_epi16(middle_dx, 1)), _mm_sub_epi16(br, bl));
                        const auto gy = _mm_add_epi16(_mm_add_epi16(_mm_sub_epi16(bl, tl),
                            _mm_slli_epi16(middle_dy, 1)), _mm_sub_epi16(br, tr));
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(gradient_x + c), gx);
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(gradient_y + c), gy);
                        const auto lo = _mm_unpacklo_epi16(gx, gy), hi = _mm_unpackhi_epi16(gx, gy);
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(norms + c), _mm_madd_epi16(lo, lo));
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(norms + c + 4), _mm_madd_epi16(hi, hi));
                    }
                    for (int pixel = 0; pixel < 8; ++pixel) {
                        const int p = 3 * pixel;
                        int best = p;
                        if (norms[p + 1] > norms[best]) best = p + 1;
                        if (norms[p + 2] > norms[best]) best = p + 2;
                        out_x[x + pixel] = float(gradient_x[best]);
                        out_y[x + pixel] = float(gradient_y[best]);
                        out_mag[x + pixel] = float(norms[best]);
                    }
                    x += 7;
                    continue;
                }
#endif
                const int left = 3 * std::max(0, x - 1);
                const int right = 3 * std::min(image.cols - 1, x + 1);
                const int center = 3 * x;
                int best = -1, best_x = 0, best_y = 0;
                for (int channel = 0; channel < 3; ++channel) {
                    const int l = left + channel, r = right + channel, c = center + channel;
                    const int gx = top[r] - top[l] + 2 * (mid[r] - mid[l]) + bottom[r] - bottom[l];
                    const int gy = bottom[l] - top[l] + 2 * (bottom[c] - top[c]) + bottom[r] - top[r];
                    const int norm = gx * gx + gy * gy;
                    if (norm > best) {
                        best = norm;
                        best_x = gx;
                        best_y = gy;
                    }
                }
                out_x[x] = float(best_x);
                out_y[x] = float(best_y);
                out_mag[x] = float(best);
            }
        }
    });
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, Mat& angle_ori, float threshold)
{
    Timer gradient_timer;
    Mat smoothed;
    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 7;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
    gradient_timer.out("gradient: blur");

    if(src.channels() == 1){
        Mat sobel_dx, sobel_dy, sobel_ag;
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        angle_ori = sobel_ag;

    }else{

        magnitude.create(src.size(), CV_32F);

        // Allocate temporary buffers
        Size size = src.size();
        Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
        Mat sobel_dy(size, CV_32F); // maximum vertical derivative
        Mat sobel_ag;               // final gradient orientation (unquantized)

        strongestGradient(smoothed, magnitude, sobel_dx, sobel_dy);
        gradient_timer.out("gradient: fused derivatives and color channel");

        // Calculate the final gradient orientations
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        gradient_timer.out("gradient: phase");
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        gradient_timer.out("gradient: histogram");
        angle_ori = sobel_ag;
    }


}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
    update();
}

void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold);
}

void ColorGradientPyramid::pyrDown()
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    ++pyramid_level;

    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;

    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }

    update();
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    dst = Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    // Want features on the border to distinguish from background
    Mat local_mask;
    if (!mask.empty())
    {
        erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
//        subtract(mask, local_mask, local_mask);
    }

    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    float threshold_sq = strong_threshold * strong_threshold;

    int nms_kernel_size = 5;
    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    for (int r = 0+nms_kernel_size/2; r < magnitude.rows-nms_kernel_size/2; ++r)
    {
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0+nms_kernel_size/2; c < magnitude.cols-nms_kernel_size/2; ++c)
        {
            if (no_mask || mask_r[c])
            {
                float score = 0;
                if(magnitude_valid.at<uchar>(r, c)>0){
                    score = magnitude.at<float>(r, c);
                    bool is_max = true;
                    for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                        for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                            if(r_offset == 0 && c_offset == 0) continue;

                            if(score < magnitude.at<float>(r+r_offset, c+c_offset)){
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                        if(!is_max) break;
                    }

                    if(is_max){
                        for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                            for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                                if(r_offset == 0 && c_offset == 0) continue;
                                magnitude_valid.at<uchar>(r+r_offset, c+c_offset) = 0;
                            }
                        }
                    }
                }

                if (score > threshold_sq && angle.at<uchar>(r, c) > 0)
                {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
                    candidates.back().f.theta = angle_ori.at<float>(r, c);
                }
            }
        }
    }
    // We require a certain number of features
    if (candidates.size() < num_features){
        if(candidates.size() <= 4) {
            return false;
        }
    }

    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);

    // selectScatteredFeatures always return true
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
    {
        return false;
    }

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

ColorGradient::ColorGradient()
    : weak_threshold(30.0f),
      num_features(63),
      strong_threshold(60.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
    : weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const
{
    return CG_NAME;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}
/****************************************************************************************\
*                                                                 Response maps                                                                                    *
\****************************************************************************************/

static void orUnaligned8u(const uchar *src, const int src_stride,
                          uchar *dst, const int dst_stride,
                          const int width, const int height)
{
    for (int r = 0; r < height; ++r)
    {
        int c = 0;

        // not aligned, which will happen because we move 1 bytes a time for spreading
        while (c < width && reinterpret_cast<uintptr_t>(src + c) % 16 != 0) {
            dst[c] |= src[c];
            c++;
        }

        // avoid out of bound when can't divid
        // note: can't use c<width !!!
        for (; c <= width-mipp::N<uint8_t>(); c+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)src + c);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst + c);

            mipp::Reg<uint8_t> res_v = mipp::orb(src_v, dst_v);
            res_v.store((uint8_t*)dst + c);
        }

        for(; c<width; c++)
            dst[c] |= src[c];

        // Advance to next row
        src += src_stride;
        dst += dst_stride;
    }
}

static void spread(const Mat &src, Mat &dst, int T)
{
    Mat horizontal = Mat::zeros(src.size(), CV_8U);
    dst = Mat::zeros(src.size(), CV_8U);
    // OR is associative: retain the original forward support and clipped border.
    for (int c = 0; c < T; ++c)
        orUnaligned8u(src.ptr() + c, static_cast<int>(src.step1()), horizontal.ptr(),
                      static_cast<int>(horizontal.step1()), src.cols - c, src.rows);
    for (int r = 0; r < T; ++r)
        orUnaligned8u(horizontal.ptr(r), static_cast<int>(horizontal.step1()), dst.ptr(),
                      static_cast<int>(dst.step1()), src.cols, src.rows - r);
}

static const unsigned char LUT3 = 3;
// 1,2-->0 3-->LUT3
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4};

static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r)
    {
        const uchar *src_r = src.ptr(r);
        uchar *lsb4_r = lsb4.ptr(r);
        uchar *msb4_r = msb4.ptr(r);

        for (int c = 0; c < src.cols; ++c)
        {
            // Least significant 4 bits of spread image pixel
            lsb4_r[c] = src_r[c] & 15;
            // Most significant 4 bits, right-shifted to be in [0, 16)
            msb4_r[c] = (src_r[c] & 240) >> 4;
        }
    }

    {
        uchar *lsb4_data = lsb4.ptr<uchar>();
        uchar *msb4_data = msb4.ptr<uchar>();

        bool no_max = true;
        bool no_shuff = true;

#ifdef has_max_int8_t
        no_max = false;
#endif

#ifdef has_shuff_int8_t
        no_shuff = false;
#endif
        // LUT is designed for 128 bits SIMD, so quite triky for others

        // For each of the 8 quantized orientations...
        for (int ori = 0; ori < 8; ++ori){
            uchar *map_data = response_maps[ori].ptr<uchar>();
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

            if(mipp::N<uint8_t>() == 1 || no_max || no_shuff){ // no SIMD
                for (int i = 0; i < src.rows * src.cols; ++i)
                    map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            }
            else if(mipp::N<uint8_t>() == 16){ // 128 SIMD, no add base

                const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
                mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
                mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
                    mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
                    mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

                    mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
                    mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

                    mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
                    result.store((uint8_t*)map_data + i);
                }
            }
            else if(mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32
                    || mipp::N<uint8_t>() == 64){ //128 256 512 SIMD
                CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);

                uint8_t lut_temp[mipp::N<uint8_t>()] = {0};

                for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
                    std::copy_n(lut_low, 16, lut_temp+slice*16);
                }
                mipp::Reg<uint8_t> lut_low_v(lut_temp);

                uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
                for(uint8_t slice=0; slice<mipp::N<uint8_t>(); slice+=16){
                    std::copy_n(lut_low+16, 16, lut_temp+slice);
                    std::fill_n(base_add_array+slice, 16, slice);
                }
                mipp::Reg<uint8_t> base_add(base_add_array);
                mipp::Reg<uint8_t> lut_high_v(lut_temp);

                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
                    mipp::Reg<uint8_t> mask_low_v((uint8_t*)lsb4_data+i);
                    mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data+i);

                    mask_low_v += base_add;
                    mask_high_v += base_add;

                    mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
                    mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

                    mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
                    result.store((uint8_t*)map_data + i);
                }
            }
            else{
                for (int i = 0; i < src.rows * src.cols; ++i)
                    map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            }
        }


    }
}

static void linearize(const Mat &response_map, Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
    {
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T)
                    *memory++ = response_data[c];
            }
        }
    }
}

static const std::array<std::array<uchar, 256>, 8>& responseLookup()
{
    static const auto lookup = [] {
        std::array<std::array<uchar, 256>, 8> table{};
        for (int ori = 0; ori < 8; ++ori)
            for (int value = 0; value < 256; ++value)
                table[ori][value] = std::max(SIMILARITY_LUT[32 * ori + (value & 15)],
                                            SIMILARITY_LUT[32 * ori + 16 + (value >> 4)]);
        return table;
    }();
    return lookup;
}

static void linearizeResponses(const Mat& spread_image, std::vector<Mat>& memories, int T)
{
    CV_Assert(spread_image.rows % T == 0 && spread_image.cols % T == 0);
    const auto& lookup = responseLookup();
    memories.resize(8);
    for (auto& memory : memories)
        memory.create(T * T, (spread_image.cols / T) * (spread_image.rows / T), CV_8U);
    parallel_for_(Range(0, 8), [&](const Range& orientations) {
        for (int ori = orientations.start; ori < orientations.end; ++ori) {
            const auto& lut = lookup[ori];
            for (int y = 0; y < T; ++y)
                for (int x = 0; x < T; ++x) {
                    uchar* output = memories[ori].ptr(y * T + x);
                    for (int r = y; r < spread_image.rows; r += T) {
                        const uchar* input = spread_image.ptr(r);
                        for (int c = x; c < spread_image.cols; c += T)
                            *output++ = lut[input[c]];
                    }
                }
        }
    });
}
/****************************************************************************************\
*                                                             Linearized similarities                                                                    *
\****************************************************************************************/

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
                                               const Feature &f, int T, int W)
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,
                       Mat &dst, Size size, int T)
{
    // we only have one modality, so 8192*2, due to mipp, back to 8192
    CV_Assert(templ.features.size() < 8192);

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    int template_positions = span_y * W + span_x + 1; // why add 1?

    dst = Mat::zeros(H, W, CV_16U);
    short *dst_ptr = dst.ptr<short>();
    mipp::Reg<uint8_t> zero_v(uint8_t(0));

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {

        Feature f = templ.features[i];

        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        int j = 0;

        // *2 to avoid int8 read out of range
        for(; j <= template_positions -mipp::N<int16_t>()*2; j+=mipp::N<int16_t>()){
            mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);

            // uchar to short, once for N bytes
            mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

            mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);

            mipp::Reg<int16_t> res_v = src16_v + dst_v;
            res_v.store((int16_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += short(lm_ptr[j]);
    }
}

static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    CV_Assert(templ.features.size() < 8192);

    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;
    mipp::Reg<uint8_t> zero_v = uint8_t(0);

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        {
            short *dst_ptr = dst.ptr<short>();

            if(mipp::N<uint8_t>() > 32){ //512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<int16_t>()/16){
                    mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row*16);

                    // load lm_ptr, 16 bytes once, for half
                    uint8_t local_v[mipp::N<uint8_t>()] = {0};
                    for(int slice=0; slice<mipp::N<uint8_t>()/16/2; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src8_v(local_v);
                    // uchar to short, once for N bytes
                    mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                    mipp::Reg<int16_t> res_v = src16_v + dst_v;
                    res_v.store((int16_t*)dst_ptr);

                    dst_ptr += mipp::N<int16_t>();
                }
            }else{ // 256 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<int16_t>()){
                        mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);

                        // uchar to short, once for N bytes
                        mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                        mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
                        mipp::Reg<int16_t> res_v = src16_v + dst_v;
                        res_v.store((int16_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

static void similarity_64(const std::vector<Mat> &linear_memories, const Template &templ,
                          Mat &dst, Size size, int T)
{
    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(templ.features.size() < 64);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    dst = Mat::zeros(H, W, CV_8U);
    uchar *dst_ptr = dst.ptr<uchar>();

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;

        for(; j <= template_positions -mipp::N<uint8_t>(); j+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

            mipp::Reg<uint8_t> res_v = src_v + dst_v;
            res_v.store((uint8_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += lm_ptr[j];
    }
}

static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
                               Mat &dst, Size size, int T, Point center)
{
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() < 64);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        {
            uchar *dst_ptr = dst.ptr<uchar>();

            if(mipp::N<uint8_t>() > 16){ // 256 or 512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<uint8_t>()/16){
                    mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr);

                    // load lm_ptr, 16 bytes once
                    uint8_t local_v[mipp::N<uint8_t>()];
                    for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src_v(local_v);

                    mipp::Reg<uint8_t> res_v = src_v + dst_v;
                    res_v.store((uint8_t*)dst_ptr);

                    dst_ptr += mipp::N<uint8_t>();
                }
            }else{ // 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<uint8_t>()){
                        mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + col);
                        mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + col);
                        mipp::Reg<uint8_t> res_v = src_v + dst_v;
                        res_v.store((uint8_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

/****************************************************************************************\
*                                                             High-level Detector API                                                                    *
\****************************************************************************************/

Detector::Detector()
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = 2;
    T_at_level.push_back(4);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T)
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, std::vector<int> T, float weak_thresh, float strong_threash)
{
    this->modality = makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
    pyramid_levels = T.size();
    T_at_level = T;
}

std::vector<Match> Detector::match(Mat source, float threshold,
                                   const std::vector<std::string> &class_ids, const Mat mask) const
{
    return matchPrepared(prepare(source, mask), threshold, class_ids);
}

static void prepareMemories(const Mat& source, const Mat& mask, int T, float weak,
                            size_t features, float strong, std::vector<Mat>& memories)
{
    ColorGradientPyramid quantizer(source, mask, weak, features, strong);
    Mat quantized, expanded;
    if (mask.empty()) quantized = quantizer.angle;
    else quantizer.quantize(quantized);
    spread(quantized, expanded, T);
    linearizeResponses(expanded, memories, T);
}

static void prepareMemoryRegion(const Mat& source, const Mat& mask, int T, float weak,
                                size_t features, float strong, Rect requested,
                                std::vector<Mat>& memories)
{
    const int x0 = std::max(0, (requested.x - 5) / 16 * 16);
    const int y0 = std::max(0, (requested.y - 5) / 16 * 16);
    const int x1 = std::min(source.cols, (requested.br().x + T + 19) / 16 * 16);
    const int y1 = std::min(source.rows, (requested.br().y + T + 19) / 16 * 16);
    const Rect patch_rect(x0, y0, x1 - x0, y1 - y0);
    const Mat patch = source(patch_rect).clone();
    const Mat patch_mask = mask.empty() ? Mat() : mask(patch_rect).clone();
    ColorGradientPyramid quantizer(patch, patch_mask, weak, features, strong);
    Mat quantized, expanded;
    if (mask.empty()) quantized = quantizer.angle;
    else quantizer.quantize(quantized);
    spread(quantized, expanded, T);
    memories.resize(8);
    const int mem_width = source.cols / T;
    for (auto& memory : memories)
        memory.create(T * T, mem_width * (source.rows / T), CV_8U);
    const auto& lookup = responseLookup();
    parallel_for_(Range(0, 8), [&](const Range& orientations) {
        for (int ori = orientations.start; ori < orientations.end; ++ori)
            for (int ry = 0; ry < T; ++ry)
                for (int rx = 0; rx < T; ++rx) {
                    const int first_y = requested.y + (ry - requested.y % T + T) % T;
                    const int first_x = requested.x + (rx - requested.x % T + T) % T;
                    for (int y = first_y; y < requested.br().y; y += T) {
                        uchar* output = memories[ori].ptr(ry * T + rx) +
                            (y / T) * mem_width + first_x / T;
                        const uchar* input = expanded.ptr(y - y0);
                        for (int x = first_x; x < requested.br().x; x += T)
                            *output++ = lookup[ori][input[x - x0]];
                    }
                }
    });
}

Detector::PreparedFrame Detector::prepare(Mat source, const Mat mask, bool eager) const
{
    return prepareImpl(std::move(source), mask, eager, true);
}

Detector::PreparedFrame Detector::prepareOwned(Mat&& source) const
{
    const bool shared = !source.u || CV_XADD(&source.u->refcount, 0) != 1;
    return prepareImpl(std::move(source), Mat(), false, shared);
}

Detector::PreparedFrame Detector::prepareImpl(Mat source, const Mat& mask, bool eager, bool snapshot) const
{
    Timer timer;
    PreparedFrame frame;
    frame.sampling = T_at_level;
    frame.levels = pyramid_levels;
    frame.weak = modality->weak_threshold;
    frame.strong = modality->strong_threshold;
    frame.features = modality->num_features;

    CV_Assert(pyramid_levels > 0 && (mask.empty() || mask.size() == source.size()));
    eager = eager || source.isSubmatrix() || mask.isSubmatrix();
    frame.storage = std::make_shared<PreparedFrame::Storage>();
    auto& storage = *frame.storage;
    storage.memories = LinearMemoryPyramid(pyramid_levels,
        std::vector<LinearMemories>(1, LinearMemories(8)));
    storage.sources.push_back(!eager && snapshot ? source.clone() : source);
    storage.masks.push_back(!eager ? mask.clone() : mask);
    for (int l = 1; l < pyramid_levels; ++l) {
        const auto& previous = storage.sources.back();
        const Size size(previous.cols / 2, previous.rows / 2);
        Mat next_source, next_mask;
        cv::pyrDown(previous, next_source, size);
        if (!mask.empty()) resize(storage.masks.back(), next_mask, size, 0, 0, INTER_NEAREST);
        storage.sources.push_back(next_source);
        storage.masks.push_back(next_mask);
    }
    for (const auto& image : storage.sources) frame.sizes.push_back(image.size());
    const int last = pyramid_levels - 1;
    prepareMemories(storage.sources[last], storage.masks[last], T_at_level[last], frame.weak,
                    frame.features >> last, frame.strong, storage.memories[last][0]);
    storage.ready = 1;
    if (eager || pyramid_levels == 1) finishPreparation(frame);
    timer.out("prepare lowest response map");
    return frame;
}

void Detector::finishPreparation(const PreparedFrame& frame, cv::Rect requested) const
{
    auto& storage = *frame.storage;
    const Rect full(Point(), frame.sizes.front());
    if (requested.empty() || frame.levels != 2 || full.width % 16 || full.height % 16 ||
        16 % frame.sampling.front()) requested = full;
    requested &= full;
    const Rect previous = storage.prepared_rect;
    const Rect expanded = previous | requested;
    if (previous == expanded) return;
    if (frame.levels != 2 || (previous.empty() && expanded == full)) {
        for (int l = 0; l < frame.levels - 1; ++l) {
            prepareMemories(storage.sources[l], storage.masks[l], frame.sampling[l], frame.weak,
                            frame.features >> l, frame.strong, storage.memories[l][0]);
        }
    } else {
        auto fill = [&](Rect area) {
            if (!area.empty()) prepareMemoryRegion(storage.sources[0], storage.masks[0],
                frame.sampling[0], frame.weak, frame.features, frame.strong, area, storage.memories[0][0]);
        };
        if (previous.empty()) fill(expanded);
        else {
            fill({expanded.x, expanded.y, expanded.width, previous.y - expanded.y});
            fill({expanded.x, previous.br().y, expanded.width, expanded.br().y - previous.br().y});
            fill({expanded.x, previous.y, previous.x - expanded.x, previous.height});
            fill({previous.br().x, previous.y, expanded.br().x - previous.br().x, previous.height});
        }
    }
    storage.prepared_rect = expanded;
    storage.ready = frame.levels;
    if (expanded == full) {
        storage.sources.clear();
        storage.masks.clear();
    }
}

std::vector<Match> Detector::matchPrepared(const PreparedFrame &frame, float threshold,
                                         const std::vector<std::string> &class_ids) const
{
    CV_Assert(frame.storage && frame.sampling == T_at_level && frame.levels == pyramid_levels &&
              frame.weak == modality->weak_threshold && frame.strong == modality->strong_threshold &&
              frame.features == modality->num_features && frame.sizes.size() == pyramid_levels);
    std::lock_guard<std::mutex> lock(frame.storage->match_mutex);
    Timer timer;
    std::vector<Match> matches;
    const auto &lm_pyramid = frame.storage->memories;
    const auto &sizes = frame.sizes;

    if (class_ids.empty())
    {
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it)
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second, &frame);
    }
    else
    {
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int)class_ids.size(); ++i)
        {
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            if (it != class_templates.end())
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second, &frame);
        }
    }

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
    matches.erase(new_end, matches.end());

    timer.out("templ match");

    return matches;
}

// Used to filter out weak matches
struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

static std::vector<Match> initialMatches(
    const std::vector<std::vector<std::vector<Mat>>>& lm_pyramid,
    const std::vector<Size>& sizes, const std::vector<int>& sampling,
    const std::vector<Template>& tp, float threshold, const std::string& class_id, size_t template_id)
{
    for (size_t level = 0; level < tp.size(); ++level) {
        const int border = level + 1 < tp.size() ? 16 * sampling[level] : 1;
        if (tp[level].width + border >= sizes[level].width ||
            tp[level].height + border >= sizes[level].height) return {};
    }
    const auto& memories = lm_pyramid.back()[0];
    const auto& templ = tp.back();
    const int T = sampling.back(), num_features = int(templ.features.size());
    Mat similarities;
    if (templ.features.size() < 64) {
        similarity_64(memories, templ, similarities, sizes.back(), T);
        similarities.convertTo(similarities, CV_16U);
    } else if (templ.features.size() < 8192) {
        similarity(memories, templ, similarities, sizes.back(), T);
    } else CV_Error(Error::StsBadArg, "feature size too large");
    std::vector<Match> candidates;
    for (int r = 0; r < similarities.rows; ++r) {
        const ushort* row = similarities.ptr<ushort>(r);
        for (int c = 0; c < similarities.cols; ++c) {
            const float score = (row[c] * 100.f) / (4 * num_features);
            if (score >= threshold) {
                const int offset = T / 2 + (T % 2 - 1);
                candidates.emplace_back(c * T + offset, r * T + offset, score,
                                        class_id, int(template_id));
            }
        }
    }
    return candidates;
}

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,
                          const std::vector<Size> &sizes,
                          float threshold, std::vector<Match> &matches,
                          const std::string &class_id,
                          const std::vector<TemplatePyramid> &template_pyramids,
                          const PreparedFrame* frame) const
{
    std::vector<std::vector<Match>> initial(template_pyramids.size());
    size_t count = 0;
    bool cached = true;
    Rect requested;
    for (size_t id = 0; id < template_pyramids.size(); ++id) {
        initial[id] = initialMatches(lm_pyramid, sizes, T_at_level, template_pyramids[id], threshold, class_id, id);
        count += initial[id].size();
        if (count > 200000) { cached = false; break; }
        if (pyramid_levels != 2 && !initial[id].empty()) requested = Rect(Point(), sizes.front());
        else for (const auto& candidate : initial[id]) {
            const auto& t = template_pyramids[id].front();
            const int T = T_at_level.front(), border = 8 * T;
            const int x = std::clamp(int(candidate.x * 2 + 1), border, sizes[0].width - t.width - border);
            const int y = std::clamp(int(candidate.y * 2 + 1), border, sizes[0].height - t.height - border);
            requested |= Rect((x / T - 8) * T, (y / T - 8) * T,
                              t.width + 16 * T + 1, t.height + 16 * T + 1) & Rect(Point(), sizes[0]);
        }
    }
    const auto adjacency = pose_adjacency.find(class_id);
    if (!requested.empty() && adjacency != pose_adjacency.end() && !adjacency->second.empty())
        requested = Rect(Point(), sizes.front());
    if (!cached) {
        initial.clear();
        if (frame) finishPreparation(*frame);
    } else {
        if (requested.empty()) return;
        if (frame) finishPreparation(*frame, requested);
    }
#pragma omp declare reduction \
    (omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp parallel for reduction(omp_insert:matches)
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
    {
        const TemplatePyramid &tp = template_pyramids[template_id];
        auto candidates = cached ? std::move(initial[template_id]) :
            initialMatches(lm_pyramid, sizes, T_at_level, tp, threshold, class_id, template_id);
        if (candidates.empty()) continue;

        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l)
        {
            const std::vector<LinearMemories> &lms = lm_pyramid[l];
            int T = T_at_level[l];
            int start = static_cast<int>(l);
            Size size = sizes[l];
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1);
            int max_x = size.width - tp[start].width - border;
            int max_y = size.height - tp[start].height - border;

            Mat similarities2;
            for (int m = 0; m < (int)candidates.size(); ++m)
            {
                Match &match2 = candidates[m];
                int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                // Compute local similarity maps for each ColorGradient
                int numFeatures = 0;

                {
                    const Template &templ = tp[start];
                    numFeatures += static_cast<int>(templ.features.size());

                    if (templ.features.size() < 64){
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                        similarities2.convertTo(similarities2, CV_16U);
                    }else if (templ.features.size() < 8192){
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                    }else{
                        CV_Error(Error::StsBadArg, "feature size too large");
                    }
                }

                // Find best local adjustment
                float best_score = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < similarities2.rows; ++r)
                {
                    ushort *row = similarities2.ptr<ushort>(r);
                    for (int c = 0; c < similarities2.cols; ++c)
                    {
                        int score_int = row[c];
                        float score = (score_int * 100.f) / (4 * numFeatures);

                        if (score > best_score)
                        {
                            best_score = score;
                            best_r = r;
                            best_c = c;
                        }
                    }
                }
                // Update current match
                match2.similarity = best_score;
                match2.x = (x / T - 8 + best_c) * T + offset;
                match2.y = (y / T - 8 + best_r) * T + offset;
                // Subpixel refinement (R08H-B5): parabolic interpolation on the
                // 3x3 response neighborhood at the finest pyramid level — the
                // integer-grid argmax is the dominant position quantization.
                if (l == 0 && best_r > 0 && best_r + 1 < similarities2.rows &&
                    best_c > 0 && best_c + 1 < similarities2.cols)
                {
                    const ushort *row_u = similarities2.ptr<ushort>(best_r - 1);
                    const ushort *row_m = similarities2.ptr<ushort>(best_r);
                    const ushort *row_d = similarities2.ptr<ushort>(best_r + 1);
                    const float c_l = float(row_m[best_c - 1]);
                    const float c_m = float(row_m[best_c]);
                    const float c_r = float(row_m[best_c + 1]);
                    const float r_u = float(row_u[best_c]);
                    const float r_d = float(row_d[best_c]);
                    const float denom_c = c_l - 2 * c_m + c_r;
                    const float denom_r = r_u - 2 * c_m + r_d;
                    const float dc = (denom_c != 0 && c_m >= c_l && c_m >= c_r)
                                         ? 0.5f * (c_l - c_r) / denom_c : 0.f;
                    const float dr = (denom_r != 0 && c_m >= r_u && c_m >= r_d)
                                         ? 0.5f * (r_u - r_d) / denom_r : 0.f;
                    match2.x = (x / T - 8 + best_c + dc) * T + offset;
                    match2.y = (y / T - 8 + best_r + dr) * T + offset;
                }
            }

            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                                  MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());

            // Subpixel angle (R08H-B6): parabolic interpolation over the
            // neighboring ANGLE poses at the survivor location, finest level.
            // Each neighbor's peak score is computed with the same 16x16 local
            // refinement used by the marching loop above (~same normalization).
            if (l == 0)
            {
                auto adj_it = pose_adjacency.find(class_id);
                if (adj_it != pose_adjacency.end() && !adj_it->second.empty())
                {
                    const auto &lms0 = lm_pyramid[0];
                    const int T0 = T_at_level[0];
                    const auto &size0 = sizes[0];
                    const int border0 = 8 * T0;
                    auto neighbor_peak = [&](int nid, int cx, int cy) -> float {
                        if (nid < 0 || nid >= (int)template_pyramids.size()) return -1.f;
                        const Template &ntempl = template_pyramids[nid][0];
                        if (ntempl.features.empty()) return -1.f;
                        int nx = std::min(std::max(cx, border0),
                                          size0.width - (int)ntempl.width - border0);
                        int ny = std::min(std::max(cy, border0),
                                          size0.height - (int)ntempl.height - border0);
                        if (nx < 0 || ny < 0) return -1.f;
                        cv::Mat nsim;
                        if (ntempl.features.size() < 64)
                        {
                            similarityLocal_64(lms0[0], ntempl, nsim, size0, T0, cv::Point(nx, ny));
                            nsim.convertTo(nsim, CV_16U);
                        }
                        else if (ntempl.features.size() < 8192)
                        {
                            similarityLocal(lms0[0], ntempl, nsim, size0, T0, cv::Point(nx, ny));
                        }
                        else
                            return -1.f;
                        const float norm = 100.f / (4.f * (float)ntempl.features.size());
                        float best = 0.f;
                        for (int r = 0; r < nsim.rows; ++r)
                        {
                            const ushort *row = nsim.ptr<ushort>(r);
                            for (int c = 0; c < nsim.cols; ++c)
                                best = std::max(best, float(row[c]) * norm);
                        }
                        return best;
                    };
                    for (Match &m : candidates)
                    {
                        const int id = m.template_id;
                        if (id < 0 || id >= (int)adj_it->second.size()) continue;
                        const int prev_id = adj_it->second[id][0];
                        const int next_id = adj_it->second[id][1];
                        if (prev_id < 0 || next_id < 0) continue;
                        const float f_p = neighbor_peak(prev_id, (int)m.x, (int)m.y);
                        const float f_n = neighbor_peak(next_id, (int)m.x, (int)m.y);
                        if (f_p < 0.f || f_n < 0.f) continue;
                        const float denom = f_p - 2.f * m.similarity + f_n;
                        if (denom != 0.f && m.similarity >= f_p && m.similarity >= f_n)
                            m.pose_offset = 0.5f * (f_p - f_n) / denom;
                    }
                }
            }
        }

        if (candidates.size() > 200000 || matches.size() > 200000 - candidates.size())
            CV_Error(Error::StsOutOfRange, "matching candidate budget exceeded");
        matches.insert(matches.end(), candidates.begin(), candidates.end());
    }
}

int Detector::addTemplate(const Mat source, const std::string &class_id,
                          const Mat &object_mask, int num_features)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    {
        // Extract a template at each pyramid level
        Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);

        if(num_features > 0)
        qp->num_features = num_features;

        for (int l = 0; l < pyramid_levels; ++l)
        {
            /// @todo Could do mask subsampling here instead of in pyrDown()
            if (l > 0)
                qp->pyrDown();

            bool success = qp->extractTemplate(tp[l]);
            if (!success)
                return -1;
        }
    }

    //    Rect bb =
    cropTemplates(tp);

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}

static cv::Point2f rotate2d(const cv::Point2f inPoint, const double angRad)
{
    cv::Point2f outPoint;
    //CW rotation
    outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
    return outPoint;
}

static cv::Point2f rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double angRad)
{
    return rotate2d(inPoint - center, angRad) + center;
}

int Detector::addTemplate_rotate(const string &class_id, int zero_id,
                                 float theta, cv::Point2f center)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    const auto& to_rotate_tp = template_pyramids[zero_id];

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    for (int l = 0; l < pyramid_levels; ++l)
    {
        if(l>0) center /= 2;

        for(auto& f: to_rotate_tp[l].features){
            Point2f p;
            p.x = f.x + to_rotate_tp[l].tl_x;
            p.y = f.y + to_rotate_tp[l].tl_y;
            Point2f p_rot = rotatePoint(p, center, -theta/180*CV_PI);

            Feature f_new;
            f_new.x = int(p_rot.x + 0.5f);
            f_new.y = int(p_rot.y + 0.5f);

            f_new.theta = f.theta - theta;
            while(f_new.theta > 360) f_new.theta -= 360;
            while(f_new.theta < 0) f_new.theta += 360;

            f_new.label = int(f_new.theta * 16 / 360 + 0.5f);
            f_new.label &= 7;


            tp[l].features.push_back(f_new);
        }
        tp[l].pyramid_level = l;
    }

    cropTemplates(tp);

    template_pyramids.push_back(tp);
    return template_id;
}
const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
    {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    modality = makePtr<ColorGradient>();
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    modality->write(fs);
}

std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty())
    {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else
    {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it)
        {
            tps[template_id][idx++].read(*templ_it);
        }
    }

    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "class_id" << it->first;
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i)
    {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "templates"
           << "[";
        for (size_t j = 0; j < tp.size(); ++j)
        {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string> &class_ids,
                           const std::string &format)
{
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        const String &class_id = class_ids[i];
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string &format) const
{
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it)
    {
        const String &class_id = it->first;
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}

} // namespace line2Dup
