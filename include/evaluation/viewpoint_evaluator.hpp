// File: evaluation/viewpoint_evaluator.hpp

#ifndef VIEWPOINT_EVALUATOR_HPP
#define VIEWPOINT_EVALUATOR_HPP

#include "cache/viewpoint_cache.hpp"
#include "common/logging/logger.hpp"
#include "common/metrics/metrics_collector.hpp"
#include "common/traits/optimization_traits.hpp"
#include "optimization/acquisition.hpp"
#include "optimization/convergence_checker.hpp"
#include "optimization/gaussian/gpr.hpp"
#include "processing/image/comparator.hpp"

template<FloatingPoint T = double, optimization::IsKernel<T> KernelType = optimization::DefaultKernel<T>>
class ViewpointEvaluator {
public:
    ViewpointEvaluator(std::shared_ptr<optimization::GPR<T, KernelType>> gpr, cache::ViewpointCache<T> &cache,
                       optimization::Acquisition<T> &acquisition,
                       optimization::ConvergenceChecker<T, KernelType> &convergence_checker) :
        gpr_(gpr), cache_(cache), acquisition_(acquisition), convergence_checker_(convergence_checker),
        current_iteration_(0) {}

    void evaluatePoint(ViewPoint<T> &point, const Image<> &target,
                       const std::shared_ptr<processing::image::ImageComparator> &comparator) {
        LOG_DEBUG("Evaluating viewpoint at position {}", point.getPosition());
        if (!point.hasScore()) {
            auto cached_score = cache_.query(point.getPosition());
            if (cached_score) {
                point.setScore(*cached_score);
                LOG_DEBUG("Using cached score {} for position {}", *cached_score, point.getPosition());
            } else {
                const Image<> rendered_image = Image<>::fromViewPoint(point);
                T score = comparator->compare(target, rendered_image);
                point.setScore(score);
                cache_.insert(point);
                LOG_DEBUG("Computed new score {} for position {}", score, point.getPosition());
            }

            gpr_->update(point.getPosition(), point.getScore());
        } else {
            // Update the cache with the existing point
            cache_.update(point);
        }

        recordMetrics(point);
        current_iteration_++;
    }

    T computeAcquisition(const Eigen::Vector3<T> &x, T mean, T std_dev) const {
        acquisition_.incrementIteration();
        return acquisition_.compute(x, mean, std_dev);
    }

    bool hasConverged(T current_score, T best_score, int current_iteration, const ViewPoint<T> &best_viewpoint) {
        return convergence_checker_.hasConverged(current_score, best_score, current_iteration, best_viewpoint, gpr_);
    }

private:
    std::shared_ptr<optimization::GPR<T, KernelType>> gpr_;
    cache::ViewpointCache<T> &cache_;
    optimization::Acquisition<T> &acquisition_;
    optimization::ConvergenceChecker<T, KernelType> &convergence_checker_;
    int current_iteration_;

    void recordMetrics(const ViewPoint<T> &point) {
        auto position = point.getPosition();

        metrics::recordMetrics(point, {{"iteration", current_iteration_},
                                       {"position_x", position.x()},
                                       {"position_y", position.y()},
                                       {"position_z", position.z()},
                                       {"score", point.getScore()}});
    }
};

#endif // VIEWPOINT_EVALUATOR_HPP
