// File: core/distance_estimator.cpp

#include "processing/vision/distance_estimator.hpp"

#include "common/utilities/camera.hpp"
#include "common/utilities/image.hpp"


double DistanceEstimator::estimateDistance(const cv::Mat &target_image,
                                           const size_t max_iterations,
                                           const double initial_distance) {
    const double target_area_ratio = calculateObjectAreaRatio(target_image);
    LOG_DEBUG("Target Image Area Ratio = {}", target_area_ratio);
    double distance = initial_distance;

    for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
        std::vector<Eigen::Matrix4d> poses = generatePoses(distance);
        double cumulative_area_ratio = 0.0;

        for (const auto &pose: poses) {
            cv::Mat rendered_image = core::Perception::render(pose);
            const double current_area_ratio = calculateObjectAreaRatio(rendered_image);
            cumulative_area_ratio += current_area_ratio;
        }

        double average_area_ratio = cumulative_area_ratio / static_cast<double>(poses.size());
        LOG_DEBUG("Iteration {}: Distance = {}, Average Area Ratio = {}", iteration, distance, average_area_ratio);

        if (std::abs(average_area_ratio - target_area_ratio) < 0.05) {
            break;
        }

        distance *= std::sqrt(target_area_ratio / average_area_ratio);
    }

    return distance;
}

double DistanceEstimator::calculateObjectAreaRatio(const cv::Mat &image) {
    const cv::Mat binary = common::utilities::toBinary(image, 128.0, 255.0);
    const double object_area = cv::countNonZero(binary);

    return object_area / static_cast<double>(image.rows * image.cols);
}

std::vector<Eigen::Matrix4d> DistanceEstimator::generatePoses(const double distance) {
    std::vector<Eigen::Matrix4d> poses;
    std::vector<std::pair<double, double> > angles = {
            {0.0, 0.0},
            {M_PI, 0.0},
            {M_PI / 2, 0.0},
            {3 * M_PI / 2, 0.0},
            {M_PI / 2, M_PI / 2},
            {M_PI / 2, 3 * M_PI / 2}
    };


    for (const auto &[theta, phi]: angles) {
        core::View view = ViewPoint<double>::fromSpherical(distance, theta, phi).toView();
        poses.push_back(view.getPose());
    }

    return poses;
}
