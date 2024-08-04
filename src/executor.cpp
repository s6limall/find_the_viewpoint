#include "executor.hpp"

#include "optimization/gpr.hpp"
#include "optimization/kernel/matern_52.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "viewpoint/evaluator.hpp"
#include "viewpoint/generator.hpp"
#include "viewpoint/octree.hpp"

using KernelType = optimization::kernel::Matern52<>;

std::once_flag Executor::init_flag_;
double Executor::radius_;
Image<> Executor::target_;
std::shared_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::shared_ptr<processing::image::FeatureMatcher> Executor::matcher_;
std::shared_ptr<processing::image::ImageComparator> Executor::comparator_;

void Executor::initialize() {
    const auto image_path = config::get("paths.target_image", Defaults::target_image_path);
    extractor_ = processing::image::FeatureExtractor::create<processing::image::AKAZEExtractor>();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();
    target_ = Image<>(common::io::image::readImage(image_path), extractor_);
    radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());

    const auto comparator_type = config::get("image.comparator.type", "SSIM");
    if (comparator_type == "SSIM") {
        LOG_INFO("Using SSIM image comparator.");
        comparator_ = std::make_shared<processing::image::SSIMComparator>();
    } else if (comparator_type == "FEATURE") {
        LOG_INFO("Using feature-based image comparator.");
        comparator_ = std::make_shared<processing::image::FeatureComparator>(extractor_, matcher_);
    } else {
        LOG_WARN("Invalid image comparator type, defaulting to SSIM.");
        comparator_ = std::make_shared<processing::image::SSIMComparator>();
    }
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    try {
        LOG_INFO("Starting viewpoint optimization.");

        // Initialize Octree
        const double size = 2 * radius_;
        const double min_size = 0.01 * size;
        viewpoint::Octree<> octree(Eigen::Vector3d::Zero(), size, min_size);

        // Initialize GPR
        const double initial_length_scale = 0.5 * size;
        const double initial_variance = 1.0;
        const double initial_noise_variance = 1e-6;
        KernelType kernel(initial_length_scale, initial_variance, initial_noise_variance);
        optimization::GaussianProcessRegression<> gpr(kernel);

        // Generate initial samples using Fibonacci lattice sampler
        FibonacciLatticeSampler<> sampler({-1, -1, -1}, {1, 1, 1}, radius_);
        const int initial_sample_count = 10; // Increased for better initial coverage
        Eigen::MatrixXd initial_samples = sampler.generate(initial_sample_count);

        // Evaluate initial samples
        ViewPoint<double> best_initial_viewpoint;
        double best_initial_score = -std::numeric_limits<double>::infinity();
        Eigen::MatrixXd X_train(initial_sample_count, 3);
        Eigen::VectorXd y_train(initial_sample_count);

        for (int i = 0; i < initial_sample_count; ++i) {
            Eigen::Vector3d position = initial_samples.col(i);
            ViewPoint<double> viewpoint(position);
            Image<> viewpoint_image = Image<>::fromViewPoint(viewpoint, extractor_);
            double score = comparator_->compare(target_, viewpoint_image);

            viewpoint.setScore(score);
            octree.evaluateAndUpdatePoint(viewpoint, target_, comparator_, gpr);

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

        // Main optimization loop
        constexpr int max_iterations = 5;
        constexpr double target_score = 0.90;
        octree.optimize(target_, comparator_, gpr, best_initial_viewpoint, max_iterations, target_score);

        // Get the final best viewpoint
        auto best_viewpoint = octree.getBestViewpoint();
        if (best_viewpoint) {
            LOG_INFO("Optimization completed. Best viewpoint: ({}, {}, {}) - Score: {}",
                     best_viewpoint->getPosition().x(), best_viewpoint->getPosition().y(),
                     best_viewpoint->getPosition().z(), best_viewpoint->getScore());
        } else {
            LOG_WARN("No suitable viewpoint found");
        }

        // Perform a final local refinement
        octree.localSearch(best_viewpoint->getPosition(), target_, comparator_, gpr);
        best_viewpoint = octree.getBestViewpoint();
        LOG_INFO("Final refined viewpoint: ({}, {}, {}) - Score: {}", best_viewpoint->getPosition().x(),
                 best_viewpoint->getPosition().y(), best_viewpoint->getPosition().z(), best_viewpoint->getScore());

    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}
