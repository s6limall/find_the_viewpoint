/*
// File: viewpoint/generator.cpp

#include "viewpoint/generator.hpp"

#include "sampling/sampler/halton_sampler.hpp"

namespace viewpoint {
    Generator::Generator(const int num_samples, const int dimensions) :
        num_samples_(num_samples), dimensions_(dimensions), estimated_distance_(0.0),
        filter_chain_(std::make_shared<filtering::HeuristicFilter>()) {
        LOG_INFO("Generator initialized with {} samples and {} dimensions.", num_samples_, dimensions_);
    }

    void Generator::setTargetImage(const cv::Mat &target_image) {
        target_image_ = target_image;
        LOG_DEBUG("Target image set with dimensions: {}x{}", target_image.cols, target_image.rows);
    }

    void Generator::setCameraIntrinsics(const core::Camera::Intrinsics &camera_intrinsics) {
        LOG_DEBUG("Setting camera parameters.");
        camera_intrinsics_ = camera_intrinsics;
    }

    double Generator::estimateDistanceToObject() {
        LOG_INFO("Starting distance estimation.");
        // processing::vision::DistanceEstimator distance_estimator(camera_intrinsics_.getFocalLengthX());
        // estimated_distance_ = distance_estimator.estimate(target_image_);
        estimated_distance_ = 1.0;
        if (estimated_distance_ <= 0) {
            LOG_ERROR("Invalid distance detected: {}", estimated_distance_);
            throw std::runtime_error("Failed to estimate distance.");
        }

        LOG_INFO("Estimated distance to object: {}", estimated_distance_);
        return estimated_distance_;
    }

    std::vector<std::vector<double> > Generator::generateInitialViewpoints(double distance) const {
        LOG_INFO("Generating initial viewpoints within spherical shell at distance {}", distance);

        constexpr double thickness_ratio = 0.5; // Adjust this for a thicker or thinner spherical shell
        sampling::HaltonSampler sampler;
        auto samples = sampler.generate(num_samples_, {0, 0, 0}, {1, 1, 1});

        LOG_INFO("Generated {} initial viewpoints", samples.size());
        return samples;
    }

    std::vector<core::View> Generator::convertToViews(const std::vector<std::vector<double> > &samples) const {
        LOG_INFO("Converting samples to views.");
        std::vector<core::View> views;
        const Eigen::Vector3d object_center(0.0, 0.0, 0.0);

        views.reserve(samples.size());
        for (const auto &sample: samples) {
            Eigen::Vector3d position(sample[0], sample[1], sample[2]);
            core::View view;
            view.computePose(position, object_center);
            views.push_back(view);
        }

        LOG_INFO("Converted {} samples to views", views.size());
        return views;
    }

    std::vector<core::View> Generator::provision() {
        LOG_INFO("Starting provision of viewpoints.");

        double distance = 0.0;
        try {
            distance = estimateDistanceToObject();
        } catch (const std::runtime_error &e) {
            LOG_ERROR("Error estimating distance: {}", e.what());
            return {};
        }

        const auto initial_samples = generateInitialViewpoints(distance);

        setupFilters();
        addHeuristics();
        const auto filtered_samples = filter_chain_->filter(initial_samples, 0.0);

        auto views = convertToViews(filtered_samples);
        LOG_INFO("Generated {} viewpoints after filtering.", views.size());

        return views;
    }

    void Generator::setupFilters() {
        // Setup the filter chain if needed
    }

    void Generator::addHeuristics() const {
        const auto distance_heuristic = std::make_shared<filtering::heuristics::DistanceHeuristic>(
                std::vector<double>{0, 0, 0});
        filter_chain_->addHeuristic(distance_heuristic, 0.5);

        auto similarity_heuristic = std::make_shared<filtering::heuristics::SimilarityHeuristic>(
                std::vector<std::vector<double> >{{1, 1, 1}});
        filter_chain_->addHeuristic(similarity_heuristic, 0.5);
    }

    void Generator::visualizeSphere(const std::string &window_name) const {
        cv::Mat display_image = target_image_.clone();
        const cv::Point center(display_image.cols / 2, display_image.rows / 2);

        const double inner_radius = estimated_distance_ * 0.9;
        const double outer_radius = estimated_distance_ * 1.1;

        cv::circle(display_image, center, static_cast<int>(inner_radius), cv::Scalar(255, 0, 0), 1);
        cv::circle(display_image, center, static_cast<int>(outer_radius), cv::Scalar(0, 255, 0), 1);

        cv::imshow(window_name, display_image);
        cv::waitKey(0);
        cv::destroyWindow(window_name);
    }

}


/*#include <spdlog/spdlog.h>
#include <Eigen/Dense>

#include "viewpoint/generator.hpp"
#include "processing/vision/scale_estimator.hpp"
#include "processing/vision/sphere_detector.hpp"
#include "sampling/halton_sampler.hpp"
#include "core/view.hpp"
#include "filtering/heuristics/distance_heuristic.hpp"
#include "filtering/heuristics/similarity_heuristic.hpp"

namespace viewpoint {
    Generator::Generator(int num_samples, int dimensions, unsigned int seed) :
        num_samples_(num_samples), dimensions_(dimensions), seed_(seed) {
        LOG_INFO("Generator initialized with {} samples, {} dimensions, and seed {}", num_samples_, dimensions_,
                     seed_);
        heuristic_filter_ = std::make_shared<filtering::HeuristicFilter>();
    }

    void Generator::setTargetImage(const cv::Mat &target_image) {
        target_image_ = target_image;
        LOG_DEBUG("Target image set with dimensions: {}x{}", target_image.cols, target_image.rows);
    }

    void Generator::setCameraMatrix(const cv::Mat &camera_matrix) {
        camera_matrix_ = camera_matrix;
        LOG_DEBUG("Camera matrix set: fx={}, fy={}, cx={}, cy={}",
                      camera_matrix_.at<double>(0, 0), camera_matrix_.at<double>(1, 1),
                      camera_matrix_.at<double>(0, 2), camera_matrix_.at<double>(1, 2));
    }

    std::pair<float, float> Generator::detectAndEstimateScaleDistance() {
        LOG_INFO("Starting sphere detection and scale/distance estimation.");
        auto sphere_detector = std::make_shared<processing::vision::SphereDetector>();
        processing::vision::ScaleEstimator scale_estimator(camera_matrix_, sphere_detector);
        auto [scale, distance] = scale_estimator.estimateScaleAndDistance(target_image_);

        if (distance <= 0 || scale <= 0) {
            LOG_ERROR("Invalid scale or distance detected: scale={}, distance={}", scale, distance);
        } else {
            LOG_INFO("Scale and distance estimation complete: scale={}, distance={}", scale, distance);
        }

        return {scale, distance};
    }

    std::vector<std::vector<double> > Generator::generateInitialViewpoints(float distance) {
        LOG_INFO("Generating initial viewpoints with distance {}", distance);

        double thickness_ratio = 0.1; // Adjust this for a thicker or thinner spherical shell
        double inner_radius = distance * (1.0 - thickness_ratio);
        double outer_radius = distance * (1.0 + thickness_ratio);

        sampling::HaltonSampler sampler;

        std::vector<double> lower_bounds = {inner_radius, 0, 0};
        std::vector<double> upper_bounds = {outer_radius, 2 * M_PI, M_PI};
        auto samples = sampler.generate(num_samples_, lower_bounds, upper_bounds, true);

        LOG_DEBUG("Generated {} initial viewpoints.", samples.size());
        return convertToCartesian(samples);
    }

    std::vector<std::vector<double> > Generator::convertToCartesian(
            const std::vector<std::vector<double> > &spherical_coords) {
        std::vector<std::vector<double> > cartesian_samples;
        cartesian_samples.reserve(spherical_coords.size());

        for (const auto &sample: spherical_coords) {
            double r = sample[0];
            double theta = sample[1];
            double phi = sample[2];
            double x = r * sin(phi) * cos(theta);
            double y = r * sin(phi) * sin(theta);
            double z = r * cos(phi);
            cartesian_samples.push_back({x, y, z});
        }

        return cartesian_samples;
    }

    std::vector<core::View> Generator::convertToViews(const std::vector<std::vector<double> > &samples) {
        LOG_INFO("Converting samples to views.");
        std::vector<core::View> views;
        Eigen::Vector3f object_center(0.0, 0.0, 0.0);

        views.reserve(samples.size());
        for (const auto &sample: samples) {
            Eigen::Vector3f position(sample[0], sample[1], sample[2]);
            core::View view;
            view.computePose(position, object_center);
            views.push_back(view);
        }
        return views;
    }

    std::vector<core::View> Generator::provision() {
        LOG_INFO("Starting provision of viewpoints.");

        auto [scale, distance] = detectAndEstimateScaleDistance();
        if (distance <= 0 || scale <= 0) {
            LOG_ERROR("Failed to estimate distance, provisioning aborted. Scale: {}, Distance: {}", scale,
                          distance);
            return {};
        }

        auto initial_samples = generateInitialViewpoints(distance);

        addHeuristics();
        auto filtered_samples = heuristic_filter_->filter(initial_samples, 0.5);

        auto views = convertToViews(filtered_samples);

        LOG_INFO("Generated {} viewpoints after filtering.", views.size());
        return views;
    }

    void Generator::addHeuristics() {
        heuristic_filter_->addHeuristic(
                std::make_shared<filtering::heuristics::DistanceHeuristic>(std::vector<double>{0, 0, 0}), 0.5);
        heuristic_filter_->addHeuristic(
                std::make_shared<filtering::heuristics::SimilarityHeuristic>(
                        std::vector<std::vector<double> >{{1, 1, 1}}), 0.5);
    }#1#

/*std::vector<core::View> Generator::provision() {
    LOG_INFO("Generating {} viewpoints...", num_samples_);

    // Define the dimensionality and bounds for the sampling
    int dimensions = 3; // Assuming 3D space (x, y, z)
    std::vector<double> lower_bounds = {-1.0, -1.0, -1.0}; // TODO: Maybe adjust later/load from config
    std::vector<double> upper_bounds = {1.0, 1.0, 1.0};

    // Create the Sampler (LHS/Halton/Sobol)
    // sampling::LHSSampler sampler(dimensions, seed_);
    sampling::HaltonSampler sampler{};

    // Generate samples with the given number of samples and bounds
    auto samples = sampler.generate(num_samples_, lower_bounds, upper_bounds, false);

    // Convert the samples into core::View objects
    std::vector<core::View> views;

    for (const auto &sample: samples) {
        // sample is of size 3 (x, y, z)
        Eigen::Vector3f position(sample[0], sample[1], sample[2]);
        core::View view;
        // Assuming a fixed object center for simplicity
        Eigen::Vector3f object_center(0.0, 0.0, 0.0);
        //view.computePose(position, Eigen::Vector3f(0, 0, 0));
        view.computePose(position.normalized() * 3.0f, object_center);
        views.push_back(view);
    }

    LOG_INFO("Generated {} viewpoints", views.size());
    return views;
}#1#
*/

