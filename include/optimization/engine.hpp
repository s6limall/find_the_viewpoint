// File: optimization/engine.hpp

#ifndef OPTIMIZATION_ENGINE_HPP
#define OPTIMIZATION_ENGINE_HPP

#include <memory>
#include <optional>
#include "config.hpp"
#include "fibonacci_lattice_sampler.hpp"
#include "logging.hpp"
#include "processing/image/feature_extractor.hpp"
#include "processing/image/image_comparator.hpp"
#include "viewpoint_optimizer.hpp"

class OptimizationEngine {
public:
    OptimizationEngine(double radius, double target_score, const Image<> &target,
                       std::shared_ptr<processing::image::ImageComparator> comparator,
                       std::shared_ptr<processing::image::FeatureExtractor> extractor) :
        radius_(radius), target_score_(target_score), target_(target), comparator_(std::move(comparator)),
        extractor_(std::move(extractor)), max_restarts_(config::get("optimization.max_restarts", 5)) {}

    std::optional<ViewPoint<>> optimize() {
        std::optional<ViewPoint<>> global_best_viewpoint;
        double global_best_score = -std::numeric_limits<double>::infinity();

        for (size_t restart_count = 1; restart_count <= max_restarts_; ++restart_count) {
            LOG_INFO("Starting optimization restart {} of {}", restart_count, max_restarts_);

            auto optimizer = createOptimizer();
            auto initial_viewpoints = generateInitialViewpoints();

            auto best_viewpoint = optimizer.optimize(target_, comparator_, initial_viewpoints);
            if (best_viewpoint && best_viewpoint->getScore() > global_best_score) {
                global_best_viewpoint = std::move(best_viewpoint);
                global_best_score = global_best_viewpoint->getScore();
            }

            if (global_best_viewpoint && (global_best_score - target_score_) >= -0.05) {
                break;
            }
        }
        return global_best_viewpoint;
    }

    size_t getMaxRestarts() const { return max_restarts_; }

private:
    struct KernelParameters {
        double length_scale;
        double variance;
        double noise_variance;
    };

    ViewpointOptimizer createOptimizer() const {
        double size = 2 * radius_;
        auto [length_scale, variance, noise_variance] = getKernelParameters(size);
        optimization::kernel::Matern52<> kernel(length_scale, variance, noise_variance);
        optimization::GaussianProcessRegression<> gpr(kernel);

        auto min_size = config::get("octree.min_size_multiplier", 0.01) * size;
        auto max_iterations = config::get("octree.max_iterations", 5);
        auto tolerance = config::get("octree.tolerance", 0.1);

        return ViewpointOptimizer(Eigen::Vector3d::Zero(), size, min_size, max_iterations, gpr, radius_, tolerance);
    }

    KernelParameters getKernelParameters(double size) const {
        return {config::get("optimization.gp.kernel.hyperparameters.length_scale", 0.5) * size,
                config::get("optimization.gp.kernel.hyperparameters.variance", 1.0),
                config::get("optimization.gp.kernel.hyperparameters.noise_variance", 1e-6)};
    }

    std::vector<ViewPoint<>> generateInitialViewpoints() const {
        FibonacciLatticeSampler<> sampler({0, 0, 0}, {1, 1, 1}, radius_);
        int sample_count = config::get("sampling.count", 20);
        auto samples = sampler.generate(sample_count);

        std::vector<ViewPoint<>> viewpoints;
        viewpoints.reserve(sample_count);

        double best_initial_score = -std::numeric_limits<double>::infinity();
        ViewPoint<> best_initial_viewpoint;

        for (int i = 0; i < sample_count; ++i) {
            Eigen::Vector3d position = samples.col(i);
            ViewPoint<> viewpoint(position);
            auto viewpoint_image = Image<>::fromViewPoint(viewpoint, extractor_);
            double score = comparator_->compare(target_, viewpoint_image);

            viewpoint.setScore(score);
            viewpoints.push_back(viewpoint);

            if (score > best_initial_score) {
                best_initial_score = score;
                best_initial_viewpoint = viewpoint;
            }

            LOG_INFO("Initial viewpoint {}: {} - Score: {}", i, viewpoint.toString(), score);
        }

        LOG_INFO("Best initial viewpoint: {} - Score: {}", best_initial_viewpoint.toString(), best_initial_score);

        return viewpoints;
    }

    double radius_;
    double target_score_;
    const Image<> &target_;
    std::shared_ptr<processing::image::ImageComparator> comparator_;
    std::shared_ptr<processing::image::FeatureExtractor> extractor_;
    size_t max_restarts_;
};


#endif // OPTIMIZATION_ENGINE_HPP
