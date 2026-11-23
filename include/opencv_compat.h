// OpenCV cross-version include shim (forced-include via CMake /FI for MSVC).
// OpenCV 5.0 moved the geometry family (getRotationMatrix2D / approxPolyDP /
// arcLength / contourArea / minAreaRect ...) from imgproc to the new geometry
// module. Vendor sources (integrity-hash pinned) include only imgproc, so the
// geometry header is force-provided here under a version guard.
// ASCII only (MSVC codepage pitfall).
#if !defined(VF_OPENCV_COMPAT_H)
#define VF_OPENCV_COMPAT_H
#if defined(__has_include)
#  if __has_include(<opencv2/geometry/2d.hpp>)
#    include <opencv2/geometry/2d.hpp>
#  endif
#  if __has_include(<opencv2/features.hpp>)
#    include <opencv2/features.hpp>
#  endif
#endif
#endif  // VF_OPENCV_COMPAT_H
