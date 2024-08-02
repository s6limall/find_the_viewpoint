// File: executor.cpp

#include "executor.hpp"

#include "common/timer.hpp"
#include "core/perception.hpp"
#include "optimization/apso.hpp"
#include "optimization/levenberg_marquardt.hpp"
#include "optimization/pso.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/image/preprocessor.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/fibonacci.hpp"

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

        optimization::AdvancedParticleSwarmOptimization<double, 3>::Parameters parameters{
                .swarm_size = config::get("optimization.apso.swarm_size", 100),
                .max_iterations = config::get("optimization.apso.max_iterations", 1000),
                .local_search_iterations = config::get("optimization.apso.local_search_iterations", 50),
                .inertia_min = config::get("optimization.apso.inertia_min", 0.4),
                .inertia_max = config::get("optimization.apso.inertia_max", 0.9),
                .cognitive_coefficient = config::get("optimization.apso.cognitive_coefficient", 2.0),
                .social_coefficient = config::get("optimization.apso.social_coefficient", 2.0),
                .velocity_max = config::get("optimization.apso.velocity_max", 0.1),
                .diversity_threshold = config::get("optimization.apso.diversity_threshold", 0.01),
                .stagnation_threshold = config::get("optimization.apso.stagnation_threshold", 20),
                .penalty_coefficient = config::get("optimization.apso.penalty_coefficient", 1e3),
                .neighborhood_size = config::get("optimization.apso.neighborhood_size", 5),
                .search_radius = config::get("optimization.apso.search_radius", radius_),
                .use_obl = config::get("optimization.apso.use_obl", true),
                .use_dynamic_topology = config::get("optimization.apso.use_dynamic_topology", true),
                .use_diversity_guided = config::get("optimization.apso.use_diversity_guided", true),
        };

        auto error_func = [&](const optimization::AdvancedParticleSwarmOptimization<double, 3>::VectorType &position) {
            const auto viewpoint = ViewPoint<>::fromCartesian(position[0], position[1], position[2]);
            const auto view = viewpoint.toView(Eigen::Vector3d::Zero());
            const Eigen::Matrix4d extrinsics = view.getPose();
            const cv::Mat rendered_view = core::Perception::render(extrinsics);
            return comparator_->compare(target_.getImage(), rendered_view);
        };

        auto jacobian_func =
                [&](const optimization::AdvancedParticleSwarmOptimization<double, 3>::VectorType &position) {
                    const double h = 1e-6; // Step size for finite differences
                    optimization::AdvancedParticleSwarmOptimization<double, 3>::JacobianType jacobian;

                    for (int i = 0; i < 3; ++i) {
                        optimization::AdvancedParticleSwarmOptimization<double, 3>::VectorType pos_plus = position;
                        optimization::AdvancedParticleSwarmOptimization<double, 3>::VectorType pos_minus = position;
                        pos_plus[i] += h;
                        pos_minus[i] -= h;

                        const double score_plus = error_func(pos_plus);
                        const double score_minus = error_func(pos_minus);

                        jacobian(0, i) = (score_plus - score_minus) / (2 * h);
                    }

                    return jacobian;
                };

        std::vector<optimization::AdvancedParticleSwarmOptimization<double, 3>::ConstraintFunction> constraints;
        constraints.emplace_back(
                [](const optimization::AdvancedParticleSwarmOptimization<double, 3>::VectorType &position) {
                    // Constraint function here, for example, ensure position within bounds
                    return 0.0;
                });

        optimization::AdvancedParticleSwarmOptimization<double, 3> apso(
                parameters, error_func, jacobian_func, constraints,
                optimization::AdvancedParticleSwarmOptimization<double, 3>::VectorType::Constant(-radius_),
                optimization::AdvancedParticleSwarmOptimization<double, 3>::VectorType::Constant(radius_));

        Timer timer("APSO Optimization");
        const Eigen::Vector3d best_position = apso.optimize();
        timer.stop();

        const auto viewpoint = ViewPoint<>::fromCartesian(best_position[0], best_position[1], best_position[2]);
        const auto view = viewpoint.toView(Eigen::Vector3d::Zero());
        const Eigen::Matrix4d extrinsics = view.getPose();
        const cv::Mat best_rendered_view = core::Perception::render(extrinsics);

        LOG_INFO("Best viewpoint: {}", best_position);

        cv::imshow("Best Viewpoint", best_rendered_view);
        cv::waitKey(0);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw e;
    }
}
