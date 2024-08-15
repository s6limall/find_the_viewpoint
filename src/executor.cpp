#include "executor.hpp"

#include "common/utilities/camera.hpp"
#include "common/utilities/visualizer.hpp"
#include "optimization/kernel/matern_52.hpp"
#include "optimization/octree.hpp"
#include "processing/image/comparison/composite_comparator.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/fibonacci.hpp"

using KernelType = optimization::kernel::Matern52<>;

std::once_flag Executor::init_flag_;
double Executor::radius_, Executor::target_score_;
Image<> Executor::target_;
std::shared_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::shared_ptr<processing::image::FeatureMatcher> Executor::matcher_;
std::shared_ptr<processing::image::ImageComparator> Executor::comparator_;

void Executor::initialize() {
    const auto image_path = config::get("paths.target_image", Defaults::target_image_path);
    loadExtractor();
    loadComparator();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();
    target_ = Image<>(common::io::image::readImage(image_path), extractor_);
    radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    try {
        LOG_INFO("Starting viewpoint optimization.");

        // Size of the optimization space / octree
        const double size = 2 * radius_;

        // Initialize GPR
        const double initial_length_scale = 0.5 * size, initial_variance = 1.0, initial_noise_variance = 1e-6;
        optimization::kernel::Matern52<> kernel(initial_length_scale, initial_variance, initial_noise_variance);
        optimization::GaussianProcessRegression gpr(kernel);

        // Initialize ViewpointOptimizer
        const double min_size = 0.01 * size;
        constexpr int max_iterations = 5; // Increase from 5 to allow for more optimization steps
        viewpoint::Octree<> octree(Eigen::Vector3d::Zero(), size, min_size, max_iterations, gpr, radius_, 0.1);


        // Generate initial samples using Fibonacci lattice sampler
        FibonacciLatticeSampler<> sampler({0, 0, 0}, {1, 1, 1}, radius_);
        const int initial_sample_count = config::get("sampling.count", 20);
        Eigen::MatrixXd initial_samples = sampler.generate(initial_sample_count);

        // Evaluate initial samples
        ViewPoint<> best_initial_viewpoint;
        double best_initial_score = -std::numeric_limits<double>::infinity();
        Eigen::MatrixXd X_train(initial_sample_count, 3);
        Eigen::VectorXd y_train(initial_sample_count);

        for (int i = 0; i < initial_sample_count; ++i) {
            Eigen::Vector3d position = initial_samples.col(i);
            ViewPoint<> viewpoint(position);
            Image<> viewpoint_image = Image<>::fromViewPoint(viewpoint, extractor_);
            double score = comparator_->compare(target_, viewpoint_image);

            viewpoint.setScore(score);

            X_train.row(i) = position.transpose();
            y_train(i) = score;

            if (score > best_initial_score) {
                best_initial_score = score;
                best_initial_viewpoint = viewpoint;
            }

            LOG_INFO("Initial viewpoint {}: ({}, {}, {}) - Score: {}", i, position.x(), position.y(), position.z(),
                     score);
        }

        LOG_INFO("Best initial viewpoint: ({}, {}, {}) - Score: {}", best_initial_viewpoint.getPosition().x(),
                 best_initial_viewpoint.getPosition().y(), best_initial_viewpoint.getPosition().z(),
                 best_initial_score);

        // Initialize GPR with all initial points
        gpr.fit(X_train, y_train);
        gpr.optimizeHyperparameters();
        // Main optimization loop
        octree.optimize(target_, comparator_, best_initial_viewpoint, target_score_);

        // Get the final best viewpoint
        auto best_viewpoint = octree.getBestViewpoint();
        if (best_viewpoint) {
            LOG_INFO("Optimization completed. Best viewpoint: ({}, {}, {}) - Score: {}",
                     best_viewpoint->getPosition().x(), best_viewpoint->getPosition().y(),
                     best_viewpoint->getPosition().z(), best_viewpoint->getScore());

            // Visualize the difference between the target image and the best viewpoint image
            Image<> best_image = Image<>::fromViewPoint(best_viewpoint.value(), extractor_);
            common::utilities::Visualizer::diff(target_, best_image);

        } else {
            LOG_WARN("No suitable viewpoint found");
        }

    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}

void Executor::loadExtractor() {
    const auto detector_type = config::get("image.detector.type", "SIFT");
    if (detector_type == "SIFT") {
        LOG_INFO("Using SIFT feature extractor.");
        extractor_ = std::make_shared<processing::image::SIFTExtractor>();
    } else if (detector_type == "AKAZE") {
        LOG_INFO("Using AKAZE feature extractor.");
        extractor_ = std::make_shared<processing::image::AKAZEExtractor>();
    } else if (detector_type == "ORB") {
        LOG_INFO("Using ORB feature extractor.");
        extractor_ = std::make_shared<processing::image::ORBExtractor>();
    } else {
        LOG_WARN("Invalid feature extractor type, defaulting to SIFT.");
        extractor_ = std::make_shared<processing::image::SIFTExtractor>();
    }
}

void Executor::loadComparator() {
    auto comparator_type = config::get("image.comparator.type", "SSIM");
    std::ranges::transform(comparator_type.begin(), comparator_type.end(), comparator_type.begin(), ::tolower);
    const std::string target_score_key = "image.comparator." + comparator_type + ".threshold";
    target_score_ = config::get(target_score_key, 0.80);
    if (comparator_type == "SSIM") {
        LOG_INFO("Using SSIM image comparator.");
        comparator_ = std::make_shared<processing::image::SSIMComparator>();
    } else if (comparator_type == "FEATURE") {
        LOG_INFO("Using feature-based image comparator.");
        comparator_ = std::make_shared<processing::image::FeatureComparator>(extractor_, matcher_);
    } else if (comparator_type == "COMPOSITE") {
        LOG_INFO("Using composite image comparator.");
        comparator_ = std::make_shared<processing::image::CompositeComparator>(extractor_, matcher_);
    } else {
        LOG_WARN("Invalid image comparator type, defaulting to SSIM.");
        comparator_ = std::make_shared<processing::image::SSIMComparator>();
    }
}
