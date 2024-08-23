#include "executor.hpp"

#include "common/utilities/camera.hpp"
#include "common/utilities/visualizer.hpp"
#include "interface/pose_callback.hpp"
#include "interface/pose_publisher.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "optimization/kernel/matern_52.hpp"
#include "optimization/octree.hpp"
#include "optimization/optimizer/gpr.hpp"
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
    if (config::get("estimation.distance.skip", true)) {
        radius_ = config::get("estimation.distance.initial_guess", 1.5);
    } else {
        radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());
    }
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    try {
        LOG_INFO("Starting viewpoint optimization.");

        const double size = 2 * radius_;
        const auto max_points = config::get("optimization.max_points", 0);

        auto pose_callback = std::make_shared<PoseCallback>();
        auto pose_publisher = std::make_shared<PosePublisher>(pose_callback);

        pose_callback->registerCallback([](const ViewPoint<> &viewpoint) {
            LOG_INFO("Received new best viewpoint: {}", viewpoint.toString());
        });

        const auto initial_length_scale = config::get("optimization.gp.kernel.matern.initial_length_scale_multiplier", 0.5) * size;
        const auto initial_variance = config::get("optimization.gp.kernel.matern.initial_variance", 1.0);
        const auto initial_noise_variance = config::get("optimization.gp.kernel.matern.initial_noise_variance", 1e-6);
        optimization::kernel::Matern52<> kernel(initial_length_scale, initial_variance, initial_noise_variance);
        optimization::GaussianProcessRegression gpr(kernel);

        FibonacciLatticeSampler<> sampler({0, 0, 0}, {1, 1, 1}, radius_);
        const int initial_sample_count = config::get("sampling.count", 20);
        Eigen::MatrixXd initial_samples = sampler.generate(initial_sample_count);

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

            LOG_INFO("Initial viewpoint {}: ({}, {}, {}) - Score: {}", i, position.x(), position.y(), position.z(), score);
        }

        LOG_INFO("Best initial viewpoint: ({}, {}, {}) - Score: {}", best_initial_viewpoint.getPosition().x(),
                 best_initial_viewpoint.getPosition().y(), best_initial_viewpoint.getPosition().z(),
                 best_initial_score);

        gpr.fit(X_train, y_train);

        const auto min_size = config::get("octree.min_size_multiplier", 0.01) * size;
        const auto max_iterations = config::get("octree.max_iterations", 5);
        const auto tolerance = config::get("octree.tolerance", 0.1);
        viewpoint::Octree<> octree(Eigen::Vector3d::Zero(), size, min_size, max_iterations, gpr, radius_, tolerance);

        size_t restart_count = 1;
        const size_t max_restarts = config::get("optimization.max_restarts", 5);
        std::optional<ViewPoint<>> best_viewpoint = best_initial_viewpoint;


        do {
            octree.optimize(target_, comparator_, best_viewpoint.value(), target_score_);
            best_viewpoint = octree.getBestViewpoint();
            ++restart_count;

            LOG_INFO("Restart {}: Best viewpoint: ({}, {}, {}) - Score: {}",
                     restart_count,
                     best_viewpoint->getPosition().x(), best_viewpoint->getPosition().y(),
                     best_viewpoint->getPosition().z(), best_viewpoint->getScore());

            // Update the search radius based on the current best score
            double current_radius = radius_ * std::exp(-5 * (best_viewpoint->getScore() - 0.5) / 0.5);
            current_radius = std::max(current_radius, min_size / 2);
            octree.setCurrentRadius(current_radius);

            LOG_INFO("Updated search radius: {}", current_radius);

        } while (max_restarts == 0 || restart_count < max_restarts && best_viewpoint && (best_viewpoint->getScore() - target_score_) > -0.05);


        if (best_viewpoint) {
            LOG_INFO("Optimization completed. Best viewpoint: ({}, {}, {}) - Score: {}",
                     best_viewpoint->getPosition().x(), best_viewpoint->getPosition().y(),
                     best_viewpoint->getPosition().z(), best_viewpoint->getScore());

            Image<> best_image = Image<>::fromViewPoint(*best_viewpoint, extractor_);
            common::utilities::Visualizer::diff(target_, best_image);

            // Visualize the search progression
            octree.visualizeSearchProgression();
        } else {
            LOG_WARN("No suitable viewpoint found");
        }

    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}

// Add this function to the Executor class to handle intermediate results:
static void handleIntermediateResult(const ViewPoint<>& viewpoint) {
    LOG_INFO("Intermediate best viewpoint: {} - Score: {}",
             viewpoint.toString(), viewpoint.getScore());
    // Here you can add code to visualize or further process intermediate results
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
