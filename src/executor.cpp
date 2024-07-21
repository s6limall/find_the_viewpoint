// File: executor.cpp

#include "executor.hpp"

#include "core/perception.hpp"
#include "processing/image/preprocessor.hpp"
#include "processing/image/ransac.hpp"
#include "processing/image/ransac_plane_detector.hpp"
#include "processing/vision/estimation/pose_estimator.hpp"
#include "sampling/sampler/halton_sampler.hpp"
#include "sampling/transformer/spherical_transformer.hpp"
#include "viewpoint/generator.hpp"


std::once_flag Executor::init_flag_;
Image<> Executor::target_;
std::unique_ptr<processing::image::ImageComparator> Executor::comparator_;
std::unique_ptr<processing::image::FeatureExtractor> Executor::extractor_;


void Executor::initialize() {
    extractor_ = FeatureExtractor::create<processing::image::AKAZEExtractor>();
    target_ = Image<>(common::io::image::readImage(Defaults::target_image_path), extractor_);
    comparator_ = std::make_unique<processing::image::SSIMComparator>(); // TODO: SSIM
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);
    try {
        // Set up camera with default parameters
        core::Camera::Intrinsics intrinsics;
        intrinsics.setIntrinsics(640, 480, 0.95, 0.75); // Default values

        // Preprocess the target image
        const auto processed_images = processing::image::Preprocessor<>::preprocess(target_.getImage());

        processing::vision::DistanceEstimator estimator;

        // Estimate the distance to the object
        double distance = estimator.estimate(target_.getImage());

        // Initialize the sampler and transformer
        sampling::SphericalShellTransformer transformer(distance * 0.9, distance * 1.1); // Example radius values
        sampling::HaltonSampler sampler(std::make_optional(
                [&transformer](const std::vector<double> &sample) { return transformer.transform(sample); }));

        std::vector<double> lower_bounds(3, 0.0);
        std::vector<double> upper_bounds(3, 1.0);

        // Generate initial viewpoints
        auto initial_viewpoints = sampler.generate(10, lower_bounds, upper_bounds);
        std::vector<ViewPoint<>> viewpoint_objects;
        for (const auto &vp: initial_viewpoints) {
            viewpoint_objects.emplace_back(vp[0], vp[1], vp[2]);
        }

        // Comparator for evaluating views
        auto comparator = std::make_unique<processing::image::SSIMComparator>();

        // Define object center
        Eigen::Vector3d object_center = Eigen::Vector3d::Zero();

        // Evaluate initial viewpoints
        for (auto &viewpoint: viewpoint_objects) {
            auto view = viewpoint.toView(object_center);
            Eigen::Matrix4d extrinsics = view.getPose();
            cv::Mat rendered_view = core::Perception::render(extrinsics);

            double score = comparator->compare(target_.getImage(), rendered_view);
            viewpoint.setScore(score);
        }

        // Sort viewpoints by score
        std::sort(viewpoint_objects.begin(), viewpoint_objects.end(),
                  [](const auto &a, const auto &b) { return a.getScore() > b.getScore(); });

        // Iterative refinement
        const int max_iterations = 10;
        for (auto &viewpoint: viewpoint_objects) {
            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                auto view = viewpoint.toView(object_center);
                Eigen::Matrix4d extrinsics = view.getPose();
                cv::Mat rendered_view = core::Perception::render(extrinsics);

                double new_score = comparator->compare(target_.getImage(), rendered_view);
                if (new_score > viewpoint.getScore()) {
                    viewpoint.setScore(new_score);
                    // Update viewpoint based on optimization technique
                } else {
                    break; // Exit loop if no improvement
                }
            }
        }

        // Final sort by SSIM score
        std::sort(viewpoint_objects.begin(), viewpoint_objects.end(),
                  [](const auto &a, const auto &b) { return a.getScore() > b.getScore(); });

        // Display the best viewpoint
        auto best_viewpoint = viewpoint_objects.front();
        auto best_view = best_viewpoint.toView(object_center);
        Eigen::Matrix4d best_extrinsics = best_view.getPose();
        cv::Mat best_rendered_view = core::Perception::render(best_extrinsics);

        LOG_INFO("Best viewpoint: {}", best_viewpoint.getPosition());

        cv::imshow("Best Viewpoint", best_rendered_view);
        cv::waitKey(0);


    } catch (const std::exception &e) {
        LOG_ERROR("Failed to initialize executor: {}", e.what());
        throw e;
    }
}
