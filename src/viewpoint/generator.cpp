// File: viewpoint/generator.cpp


#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include "viewpoint/generator.hpp"
#include "processing/vision/distance_estimator.hpp"
#include "sampling/constrained_spherical_sampler.hpp"
#include "filtering/heuristics/distance_heuristic.hpp"
#include "filtering/heuristics/similarity_heuristic.hpp"

namespace viewpoint {

    Generator::Generator(int num_samples, int dimensions) :
        num_samples_(num_samples), dimensions_(dimensions), estimated_distance_(0.0) {
        spdlog::info("Generator initialized with {} samples and {} dimensions.", num_samples_, dimensions_);
        filter_chain_ = std::make_shared<filtering::HeuristicFilter>();
    }

    void Generator::setTargetImage(const cv::Mat &target_image) {
        target_image_ = target_image;
        spdlog::debug("Target image set with dimensions: {}x{}", target_image.cols, target_image.rows);
    }

    void Generator::setCameraMatrix(const cv::Mat &camera_matrix) {
        camera_matrix_ = camera_matrix;
        spdlog::debug("Camera matrix set: fx={}, fy={}, cx={}, cy={}",
                      camera_matrix_.at<double>(0, 0), camera_matrix_.at<double>(1, 1),
                      camera_matrix_.at<double>(0, 2), camera_matrix_.at<double>(1, 2));
    }

    double Generator::estimateDistanceToObject() {
        spdlog::info("Starting distance estimation.");
        processing::vision::DistanceEstimator distance_estimator(1.0, target_image_.cols,
                                                                 camera_matrix_.at<double>(0, 0));
        estimated_distance_ = distance_estimator.estimate(target_image_);

        if (estimated_distance_ <= 0) {
            spdlog::error("Invalid distance detected: {}", estimated_distance_);
            throw std::runtime_error("Failed to estimate distance.");
        }

        spdlog::info("Estimated distance to object: {}", estimated_distance_);
        return estimated_distance_;
    }

    std::vector<std::vector<double> > Generator::generateInitialViewpoints(double distance) const {
        spdlog::info("Generating initial viewpoints within spherical shell at distance {}", distance);

        double thickness_ratio = 0.1; // Adjust this for a thicker or thinner spherical shell
        sampling::ConstrainedSphericalSampler sampler(distance * (1.0 - thickness_ratio),
                                                      distance * (1.0 + thickness_ratio));
        auto samples = sampler.generate(num_samples_, {}, {}, false);

        spdlog::info("Generated {} initial viewpoints", samples.size());
        return samples;
    }

    std::vector<core::View> Generator::convertToViews(const std::vector<std::vector<double> > &samples) const {
        spdlog::info("Converting samples to views.");
        std::vector<core::View> views;
        Eigen::Vector3f object_center(0.0, 0.0, 0.0);

        views.reserve(samples.size());
        for (const auto &sample: samples) {
            Eigen::Vector3f position(sample[0], sample[1], sample[2]);
            core::View view;
            view.computePoseFromPositionAndObjectCenter(position, object_center);
            views.push_back(view);
        }

        spdlog::info("Converted {} samples to views", views.size());
        return views;
    }

    std::vector<core::View> Generator::provision() {
        spdlog::info("Starting provision of viewpoints.");

        double distance = 0.0;
        try {
            distance = estimateDistanceToObject();
        } catch (const std::runtime_error &e) {
            spdlog::error("Error estimating distance: {}", e.what());
            return {};
        }

        auto initial_samples = generateInitialViewpoints(distance);

        setupFilters();
        addHeuristics();
        auto filtered_samples = filter_chain_->filter(initial_samples, 0.5);

        auto views = convertToViews(filtered_samples);
        spdlog::info("Generated {} viewpoints after filtering.", views.size());

        return views;
    }

    void Generator::setupFilters() {
        // Setup the filter chain if needed
    }

    void Generator::addHeuristics() {
        auto distance_heuristic = std::make_shared<filtering::heuristics::DistanceHeuristic>(
                std::vector<double>{0, 0, 0});
        filter_chain_->addHeuristic(distance_heuristic, 0.5);

        auto similarity_heuristic = std::make_shared<filtering::heuristics::SimilarityHeuristic>(
                std::vector<std::vector<double> >{{1, 1, 1}});
        filter_chain_->addHeuristic(similarity_heuristic, 0.5);
    }

    void Generator::visualizeSphere(const std::string &window_name) const {
        cv::Mat display_image = target_image_.clone();
        cv::Point center(display_image.cols / 2, display_image.rows / 2);

        double inner_radius = estimated_distance_ * 0.9;
        double outer_radius = estimated_distance_ * 1.1;

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
        spdlog::info("Generator initialized with {} samples, {} dimensions, and seed {}", num_samples_, dimensions_,
                     seed_);
        heuristic_filter_ = std::make_shared<filtering::HeuristicFilter>();
    }

    void Generator::setTargetImage(const cv::Mat &target_image) {
        target_image_ = target_image;
        spdlog::debug("Target image set with dimensions: {}x{}", target_image.cols, target_image.rows);
    }

    void Generator::setCameraMatrix(const cv::Mat &camera_matrix) {
        camera_matrix_ = camera_matrix;
        spdlog::debug("Camera matrix set: fx={}, fy={}, cx={}, cy={}",
                      camera_matrix_.at<double>(0, 0), camera_matrix_.at<double>(1, 1),
                      camera_matrix_.at<double>(0, 2), camera_matrix_.at<double>(1, 2));
    }

    std::pair<float, float> Generator::detectAndEstimateScaleDistance() {
        spdlog::info("Starting sphere detection and scale/distance estimation.");
        auto sphere_detector = std::make_shared<processing::vision::SphereDetector>();
        processing::vision::ScaleEstimator scale_estimator(camera_matrix_, sphere_detector);
        auto [scale, distance] = scale_estimator.estimateScaleAndDistance(target_image_);

        if (distance <= 0 || scale <= 0) {
            spdlog::error("Invalid scale or distance detected: scale={}, distance={}", scale, distance);
        } else {
            spdlog::info("Scale and distance estimation complete: scale={}, distance={}", scale, distance);
        }

        return {scale, distance};
    }

    std::vector<std::vector<double> > Generator::generateInitialViewpoints(float distance) {
        spdlog::info("Generating initial viewpoints with distance {}", distance);

        double thickness_ratio = 0.1; // Adjust this for a thicker or thinner spherical shell
        double inner_radius = distance * (1.0 - thickness_ratio);
        double outer_radius = distance * (1.0 + thickness_ratio);

        sampling::HaltonSampler sampler;

        std::vector<double> lower_bounds = {inner_radius, 0, 0};
        std::vector<double> upper_bounds = {outer_radius, 2 * M_PI, M_PI};
        auto samples = sampler.generate(num_samples_, lower_bounds, upper_bounds, true);

        spdlog::debug("Generated {} initial viewpoints.", samples.size());
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
        spdlog::info("Converting samples to views.");
        std::vector<core::View> views;
        Eigen::Vector3f object_center(0.0, 0.0, 0.0);

        views.reserve(samples.size());
        for (const auto &sample: samples) {
            Eigen::Vector3f position(sample[0], sample[1], sample[2]);
            core::View view;
            view.computePoseFromPositionAndObjectCenter(position, object_center);
            views.push_back(view);
        }
        return views;
    }

    std::vector<core::View> Generator::provision() {
        spdlog::info("Starting provision of viewpoints.");

        auto [scale, distance] = detectAndEstimateScaleDistance();
        if (distance <= 0 || scale <= 0) {
            spdlog::error("Failed to estimate distance, provisioning aborted. Scale: {}, Distance: {}", scale,
                          distance);
            return {};
        }

        auto initial_samples = generateInitialViewpoints(distance);

        addHeuristics();
        auto filtered_samples = heuristic_filter_->filter(initial_samples, 0.5);

        auto views = convertToViews(filtered_samples);

        spdlog::info("Generated {} viewpoints after filtering.", views.size());
        return views;
    }

    void Generator::addHeuristics() {
        heuristic_filter_->addHeuristic(
                std::make_shared<filtering::heuristics::DistanceHeuristic>(std::vector<double>{0, 0, 0}), 0.5);
        heuristic_filter_->addHeuristic(
                std::make_shared<filtering::heuristics::SimilarityHeuristic>(
                        std::vector<std::vector<double> >{{1, 1, 1}}), 0.5);
    }*/

/*std::vector<core::View> Generator::provision() {
    spdlog::info("Generating {} viewpoints...", num_samples_);

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
        //view.computePoseFromPositionAndObjectCenter(position, Eigen::Vector3f(0, 0, 0));
        view.computePoseFromPositionAndObjectCenter(position.normalized() * 3.0f, object_center);
        views.push_back(view);
    }

    spdlog::info("Generated {} viewpoints", views.size());
    return views;
}*/

