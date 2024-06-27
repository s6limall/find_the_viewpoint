// File: processing/vision/estimator.hpp

#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <opencv2/core.hpp>
#include <vector>

namespace processing::vision {
    class Estimator {
    public:
        virtual ~Estimator() = default;
        virtual double estimate(const cv::Mat& image) = 0;
    };
}

#endif //ESTIMATOR_HPP
