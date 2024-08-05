#include "executor.hpp"

#include "../include/optimization/octree.hpp"
#include "optimization/gpr.hpp"
#include "optimization/kernel/matern_52.hpp"
#include "processing/image/comparison/composite_comparator.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "viewpoint/evaluator.hpp"
#include "viewpoint/generator.hpp"

using KernelType = optimization::kernel::Matern52<>;

std::once_flag Executor::init_flag_;
double Executor::radius_, Executor::target_score_;
Image<> Executor::target_;
std::shared_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::shared_ptr<processing::image::FeatureMatcher> Executor::matcher_;
std::shared_ptr<processing::image::ImageComparator> Executor::comparator_;

void Executor::initialize() {
    const auto image_path = config::get("paths.target_image", Defaults::target_image_path);
    extractor_ = fetchDetector();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();
    target_ = Image<>(common::io::image::readImage(image_path), extractor_);
    radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());

    const auto comparator_type = config::get("image.comparator.type", "SSIM");
    if (comparator_type == "SSIM") {
        LOG_INFO("Using SSIM image comparator.");
        target_score_ = 0.95;
        comparator_ = std::make_shared<processing::image::SSIMComparator>();
    } else if (comparator_type == "FEATURE") {
        LOG_INFO("Using feature-based image comparator.");
        target_score_ = 0.60;
        comparator_ = std::make_shared<processing::image::FeatureComparator>(extractor_, matcher_);
    } else if (comparator_type == "COMPOSITE") {
        LOG_INFO("Using composite image comparator.");
        target_score_ = 0.775;
        comparator_ = std::make_shared<processing::image::CompositeComparator>(extractor_, matcher_);
    } else {
        LOG_WARN("Invalid image comparator type, defaulting to SSIM.");
        target_score_ = 0.95;
        comparator_ = std::make_shared<processing::image::SSIMComparator>();
    }
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

        // Initialize Octree
        const double min_size = 0.01 * size;
        constexpr int max_iterations = 5; // Increase from 5 to allow for more optimization steps
        viewpoint::Octree<double> octree(Eigen::Vector3d::Zero(), size, min_size, max_iterations, gpr, radius_, 0.1);


        // Generate initial samples using Fibonacci lattice sampler
        FibonacciLatticeSampler<> sampler({0, 0, 0}, {1, 1, 1}, radius_);
        const int initial_sample_count = config::get("sampling.count", 20);
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
        octree.optimize(target_, comparator_, best_initial_viewpoint, target_score_);

        // Get the final best viewpoint
        auto best_viewpoint = octree.getBestViewpoint();
        if (best_viewpoint) {
            LOG_INFO("Optimization completed. Best viewpoint: ({}, {}, {}) - Score: {}",
                     best_viewpoint->getPosition().x(), best_viewpoint->getPosition().y(),
                     best_viewpoint->getPosition().z(), best_viewpoint->getScore());

            /*
             * Visualize the target and the best viewpoints
             */

            // Generate image for the best viewpoint
            Image<> best_image = Image<>::fromViewPoint(best_viewpoint.value(), extractor_);

            cv::Mat b = best_image.getImage();
            cv::Mat t = target_.getImage();

            // Calculate dimensions
            int gap = 10;
            int width = t.cols + b.cols + gap;
            int height = std::max(t.rows, b.rows);

            // Create a blank image with a white background
            cv::Mat display(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

            // Copy the images into the display
            t.copyTo(display(cv::Rect(0, 0, t.cols, t.rows)));
            b.copyTo(display(cv::Rect(t.cols + gap, 0, b.cols, b.rows)));

            // Add labels
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.8;
            int thickness = 2;
            cv::Scalar text_color(0, 0, 0); // Black text

            cv::putText(display, "Target Image", cv::Point(10, 30), font_face, font_scale, text_color, thickness);
            cv::putText(display, "Best Viewpoint", cv::Point(t.cols + gap + 10, 30), font_face, font_scale, text_color,
                        thickness);

            // Display the combined image
            cv::namedWindow("Target vs Best Viewpoint", cv::WINDOW_NORMAL);
            cv::imshow("Target vs Best Viewpoint", display);
            cv::waitKey(0);
            cv::destroyAllWindows();

        } else {
            LOG_WARN("No suitable viewpoint found");
        }

    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}

std::shared_ptr<processing::image::FeatureExtractor> Executor::fetchDetector() {
    const auto detector_type = config::get("image.detector.type", "SIFT");
    if (detector_type == "SIFT") {
        LOG_INFO("Using SIFT feature extractor.");
        return std::make_shared<processing::image::SIFTExtractor>();
    } else if (detector_type == "AKAZE") {
        LOG_INFO("Using AKAZE feature extractor.");
        return std::make_shared<processing::image::AKAZEExtractor>();
    } else if (detector_type == "ORB") {
        LOG_INFO("Using ORB feature extractor.");
        return std::make_shared<processing::image::ORBExtractor>();
    } else {
        LOG_WARN("Invalid feature extractor type, defaulting to SIFT.");
        return std::make_shared<processing::image::SIFTExtractor>();
    }
}
