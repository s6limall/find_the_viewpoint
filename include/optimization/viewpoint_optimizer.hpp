// File: optimization/viewpoint_optimizer.hpp

#ifndef VIEWPOINT_OPTIMIZER_HPP
#define VIEWPOINT_OPTIMIZER_HPP

#include <optional>
#include "common/logging/logger.hpp"
#include "common/metrics/metrics_collector.hpp"
#include "common/traits/optimization_traits.hpp"
#include "optimization/convergence_checker.hpp"
#include "optimization/engine.hpp"
#include "optimization/local_refiner.hpp"
#include "optimization/radius_optimizer.hpp"

namespace optimization {

    template<FloatingPoint T = double, IsKernel<T> KernelType = DefaultKernel<T>>
    class ViewpointOptimizer {
    public:
        ViewpointOptimizer(const Eigen::Vector3<T> &center, T size, T min_size, const int max_iterations,
                           std::shared_ptr<GPR<T, KernelType>> gpr,
                           std::shared_ptr<processing::image::ImageComparator> comparator,
                           std::optional<T> radius = std::nullopt, std::optional<T> tolerance = std::nullopt) :
            engine_(center, size, min_size, gpr, radius, tolerance), convergence_checker_(),
            local_refiner_(comparator, gpr), gpr_(gpr), max_iterations_(max_iterations),
            refinement_threshold_(config::get("optimization.refinement_threshold", T(0.8))),
            significant_improvement_threshold_(config::get("optimization.significant_improvement_threshold", T(0.01))),
            uncertainty_threshold_(config::get("optimization.uncertainty_threshold", T(0.1))),
            max_local_refinement_steps_(config::get("optimization.local_refinement.patience", 5)) {}

        void optimize(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                      const ViewPoint<T> &initial_best, T target_score = T(0.95)) {
            LOG_INFO("Starting optimization with target score {}", target_score);
            convergence_checker_.setTargetScore(target_score);
            best_viewpoint_ = initial_best;
            T best_score = initial_best.getScore();

            const auto hyperparameter_optimization_frequency =
                    config::get("optimization.gp.kernel.hyperparameters.optimization.frequency", 10);

            int local_refinement_steps = 0;

            for (int i = 0; i < max_iterations_; ++i) {
                if (state::get("count", 0) > config::get("optimization.max_points", 0)) {
                    LOG_INFO("Maximum number of points reached. Stopping optimization.");
                    break;
                }

                if (state::get("count", 0) % config::get("optimization.local_refinement.frequency", 10) == 0) {
                    localRefinement(target, comparator);
                } else {
                    auto refined_viewpoint = engine_.refine(target, comparator, i);

                    if (refined_viewpoint) {
                        T current_score = refined_viewpoint->getScore();
                        engine_.addRecentScore(current_score);

                        if (current_score > best_score) {
                            T improvement = current_score - best_score;
                            best_score = current_score;
                            best_viewpoint_ = *refined_viewpoint;

                            if (improvement > significant_improvement_threshold_) {
                                localRefinement(target, comparator);
                            }
                        }

                        // Check acquisition function uncertainty and mean score
                        auto [mean, uncertainty] = gpr_->predict(refined_viewpoint->getPosition());
                        if (mean >= refinement_threshold_ && uncertainty <= uncertainty_threshold_) {
                            LOG_INFO("Trying local refinement due to low uncertainty and high mean score.");
                            localRefinement(target, comparator);
                        }

                    } else {
                        LOG_INFO("Refinement complete at iteration {}", i);
                        break;
                    }
                }

                if (convergence_checker_.hasConverged(best_score, best_score, i, best_viewpoint_, gpr_)) {
                    LOG_INFO("Optimization converged at iteration {}", i);
                    break;
                }

                if (i % hyperparameter_optimization_frequency == 0) {
                    gpr_->optimizeHyperparameters();
                }
            }

            auto refined_result = optimizeRadius(target);
            Image<T> refined_image = Image<>::fromViewPoint(refined_result->best_viewpoint);
            T refined_score = comparator->compare(target, refined_image);
            refined_result->best_viewpoint.setScore(refined_score);
            refined_image.setScore(refined_score);

            if (refined_score > best_viewpoint_.getScore()) {
                best_viewpoint_ = refined_result->best_viewpoint;
                LOG_INFO("Radius refinement improved viewpoint, best viewpoint: {}", best_viewpoint_.toString());
            }

            LOG_INFO("Optimization complete.");
            LOG_INFO("Initial best viewpoint: {}", initial_best.toString());
            LOG_INFO("Best viewpoint after main optimization: {}", best_viewpoint_.toString());
        }


        [[nodiscard]] std::optional<ViewPoint<T>> getBestViewpoint() const noexcept { return best_viewpoint_; }

    private:
        OptimizationEngine<T> engine_;
        ConvergenceChecker<T> convergence_checker_;
        LocalRefiner<T> local_refiner_;
        std::shared_ptr<GPR<T, KernelType>> gpr_;
        ViewPoint<T> best_viewpoint_;
        int max_iterations_;
        T refinement_threshold_;
        T significant_improvement_threshold_;
        T uncertainty_threshold_;
        int max_local_refinement_steps_;

        void localRefinement(const Image<> &target, std::shared_ptr<processing::image::ImageComparator> comparator) {
            ViewPoint<T> refined_viewpoint = local_refiner_.refine(target, best_viewpoint_);

            if (refined_viewpoint.getScore() > best_viewpoint_.getScore()) {
                best_viewpoint_ = refined_viewpoint;
                LOG_INFO("Local refinement improved viewpoint: {}", best_viewpoint_.toString());
            }
        }

        std::optional<typename RadiusOptimizer<T>::RadiusOptimizerResult> optimizeRadius(const Image<> &target) const {
            auto renderFunction = [](const ViewPoint<T> &vp) { return Image<T>::fromViewPoint(vp); };

            auto radius_optimizer = RadiusOptimizer<T>();
            auto result = radius_optimizer.optimize(best_viewpoint_, target, renderFunction);

            LOG_INFO("Final radius refinement complete.");

            return result;
        }
    };

} // namespace optimization

#endif // VIEWPOINT_OPTIMIZER_HPP
