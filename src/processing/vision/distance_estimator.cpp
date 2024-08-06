// File: core/distance_estimator.cpp

#include "processing/vision/estimation/distance_estimator.hpp"

#include "common/utilities/camera.hpp"
#include "common/utilities/image.hpp"

namespace processing::vision {

    double DistanceEstimator::estimate(const cv::Mat &image) const noexcept {
        const auto initial_distance = config::get("estimation.distance.initial_guess", 1.0);
        const auto max_iterations = config::get("estimation.distance.max_iterations", 10);
        const auto threshold = config::get("estimation.distance.threshold", 0.05);

        return estimateDistance(image, max_iterations, initial_distance, threshold);
    }

    double DistanceEstimator::estimateDistance(const cv::Mat &target_image, const size_t max_iterations,
                                               const double initial_distance, const double threshold) {
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

            if (const auto score = std::abs(average_area_ratio - target_area_ratio); score < threshold) {
                LOG_DEBUG("Convergence check: Score = {}, Threshold = {}, {}", score, threshold,
                          score < threshold ? "score < threshold!" : "score >= threshold!");
                LOG_DEBUG("Converged after {} iterations, with distance = {}", iteration, distance);
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
        constexpr std::array<std::pair<double, double>, 4> angles = {{
                {0.0, 0.0}, // Front
                {M_PI, 0.0}, // Back
                {M_PI / 2, 0.0}, // Left
                {3 * M_PI / 2, 0.0}, // Right
        }};

        std::vector<Eigen::Matrix4d> poses;

        for (const auto &[theta, phi]: angles) {
            core::View view = ViewPoint<>::fromSpherical(distance, theta, phi).toView();
            poses.push_back(view.getPose());
        }

        return poses;
    }

} // namespace processing::vision
