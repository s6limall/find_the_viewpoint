// File: processing/vision/distance_estimator.hpp

#ifndef DISTANCE_ESTIMATOR_HPP
#define DISTANCE_ESTIMATOR_HPP

#include <opencv2/core.hpp>
#include "processing/image/feature/extractor.hpp"
#include "processing/vision/estimator.hpp"

namespace processing::vision {

    class DistanceEstimator final : public Estimator {
    public:
        explicit DistanceEstimator(float focal_length, float unit_cube_size = 1.0);

        double estimate(const cv::Mat &image) override;

    private:
        double unit_cube_size_;
        double focal_length_;

        std::unique_ptr<image::FeatureExtractor> feature_extractor_;

        static double calculateAverageKeypointSize(const std::vector<cv::KeyPoint> &keypoints);
    };

}

#endif //DISTANCE_ESTIMATOR_HPP
