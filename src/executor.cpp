// File: executor.cpp

#include "executor.hpp"

#include "common/timer.hpp"
#include "core/perception.hpp"
#include "optimization/cmaes.hpp"
#include "optimization/pso.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/image/preprocessor.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/fibonacci.hpp"
#include "sampling/sampler/halton.hpp"

std::once_flag Executor::init_flag_;
double Executor::radius_;
Image<> Executor::target_;
std::shared_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::shared_ptr<processing::image::FeatureMatcher> Executor::matcher_;
std::shared_ptr<processing::image::ImageComparator> Executor::comparator_;

void Executor::initialize() {
    extractor_ = processing::image::FeatureExtractor::create<processing::image::AKAZEExtractor>();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();
    target_ = Image<>(common::io::image::readImage(Defaults::target_image_path), extractor_);
    radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());
    comparator_ = std::make_shared<processing::image::FeatureComparator>(extractor_, matcher_);

    // comparator_ = std::make_shared<processing::image::SSIMComparator>();
}


void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);
    try {

        const ParticleSwarmOptimization::Parameters parameters{
                .swarm_size = config::get("optimization.pso.swarm_size", 20),
                .max_iterations = config::get("optimization.pso.max_iterations", 20),
                .local_search_iterations = config::get("optimization.pso.local_search_iterations", 10),
                .inertia_weight = config::get("optimization.pso.inertia_weight", 0.5),
                .cognitive_coefficient = config::get("optimization.pso.cognitive_coefficient", 1.5),
                .social_coefficient = config::get("optimization.pso.social_coefficient", 1.5),
                .radius = radius_,
                .tolerance = config::get("optimization.pso.tolerance", 0.2),
                .inertia_min = config::get("optimization.pso.inertia_min", 0.5),
                .inertia_max = config::get("optimization.pso.inertia_max", 0.9),
                .early_termination_window = config::get("optimization.pso.early_termination_window", 10),
                .early_termination_threshold = config::get("optimization.pso.early_termination_threshold", 0.1),
                .velocity_max = config::get("optimization.pso.velocity_max", 1.0),
        };

        ParticleSwarmOptimization pso(parameters, target_, comparator_);

        Timer timer("PSO Optimization");
        const ViewPoint<> best = pso.optimize();
        timer.stop();

        const cv::Mat best_rendered_view = core::Perception::render(best.toView().getPose());

        LOG_INFO("Best viewpoint: {}", best.getPosition());

        cv::imshow("Best Viewpoint", best_rendered_view);
        cv::waitKey(0);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw e;
    }
}
