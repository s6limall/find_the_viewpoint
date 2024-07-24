// File: executor.cpp

#include "executor.hpp"
#include "core/perception.hpp"
#include "optimization/cmaes.hpp"
#include "optimization/pso.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/image/preprocessor.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/halton.hpp"

std::once_flag Executor::init_flag_;
double Executor::radius_;
Image<> Executor::target_;
std::unique_ptr<processing::image::ImageComparator> Executor::comparator_;
std::unique_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::unique_ptr<processing::image::FeatureMatcher> Executor::matcher_;


void Executor::initialize() {
    extractor_ = FeatureExtractor::create<processing::image::AKAZEExtractor>();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();
    target_ = Image<>(common::io::image::readImage(Defaults::target_image_path), extractor_);
    radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());
    comparator_ = std::make_unique<processing::image::FeatureComparator>(std::move(extractor_), std::move(matcher_));
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);
    try {
        const ParticleSwarmOptimization::Parameters pso_params{20, 20, 0.7, 1.5, 1.5, radius_, 0.3}; // Example values
        ParticleSwarmOptimization pso(pso_params, target_, std::make_unique<processing::image::SSIMComparator>());

        const ViewPoint<> best_viewpoint = pso.optimize();
        const auto best_view = best_viewpoint.toView(Eigen::Vector3d::Zero());
        const Eigen::Matrix4d best_extrinsics = best_view.getPose();
        const cv::Mat best_rendered_view = core::Perception::render(best_extrinsics);

        LOG_INFO("Best viewpoint: {}", best_viewpoint.getPosition());

        cv::imshow("Best Viewpoint", best_rendered_view);
        cv::waitKey(0);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw e;
    }
}
