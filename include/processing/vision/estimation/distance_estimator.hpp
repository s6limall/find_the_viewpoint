// File: processing/vision/estimation/distance_estimator.hpp

#ifndef DISTANCE_ESTIMATOR_HPP
#define DISTANCE_ESTIMATOR_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "core/view.hpp"
#include "core/perception.hpp"
#include "types/viewpoint.hpp"
#include "common/io/image.hpp"
#include "processing/image_processor.hpp"
#include "common/logging/logger.hpp"
#include "processing/vision/estimator.hpp"

namespace processing::vision {
    class DistanceEstimator final : public Estimator<> {
    public:
        [[nodiscard]] double estimate(const cv::Mat &image) const noexcept override;

    private:
        static double calculateObjectAreaRatio(const cv::Mat &image);

        static std::vector<Eigen::Matrix4d> generatePoses(double distance);

        static double estimateDistance(const cv::Mat &target_image, size_t max_iterations = 10,
                                       double initial_distance = 1.0, double threshold = 0.05);
    };
}

#endif // DISTANCE_ESTIMATOR_HPP
