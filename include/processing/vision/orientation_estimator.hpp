// File: processing/vision/orientation_estimator.hpp

#ifndef ORIENTATION_ESTIMATOR_HPP
#define ORIENTATION_ESTIMATOR_HPP
#include "processing/vision/estimator.hpp"

namespace processing::vision {
    class OrientationEstimator final : public Estimator {
    public:
        OrientationEstimator() = default;

        ~OrientationEstimator() override = default;

        double estimate(const cv::Mat &image) override;
    };
}

#endif //ORIENTATION_ESTIMATOR_HPP
