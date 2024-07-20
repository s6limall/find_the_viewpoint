// File: processing/vision/estimator.hpp

#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>

namespace processing::vision {
    template<typename T = double>
    class Estimator {
    public:
        virtual ~Estimator() = default;

        virtual auto estimate(const cv::Mat &image) const noexcept -> T = 0;
    };
} // namespace processing::vision

#endif // ESTIMATOR_HPP
