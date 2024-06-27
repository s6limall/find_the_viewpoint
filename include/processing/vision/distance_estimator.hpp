// File: processing/vision/distance_estimator.hpp

#ifndef DISTANCE_ESTIMATOR_HPP
#define DISTANCE_ESTIMATOR_HPP

#include <opencv2/core.hpp>
#include "processing/image/feature/extractor.hpp"
#include "processing/vision/estimator.hpp"

namespace processing::vision {

    class DistanceEstimator : public Estimator {
    public:
        explicit DistanceEstimator(double unit_cube_size = 1.0, int image_width = 640, double fov_x = 60.0);

        double estimate(const cv::Mat &image) override;

    private:
        double focal_length_;
        double unit_cube_size_;
        std::unique_ptr<image::FeatureExtractor> feature_extractor_;

        static double calculateAverageKeypointSize(const std::vector<cv::KeyPoint> &keypoints);

        static double calculateFocalLength(int image_width, double fov_x);
    };

}

#endif //DISTANCE_ESTIMATOR_HPP
