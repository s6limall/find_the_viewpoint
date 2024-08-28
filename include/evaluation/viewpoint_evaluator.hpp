// File: evaluation/viewpoint_evaluator.hpp

#ifndef VIEWPOINT_EVALUATOR_HPP
#define VIEWPOINT_EVALUATOR_HPP
#include "cache/viewpoint_cache.hpp"
#include "common/logging/logger.hpp"
#include "optimization/acquisition.hpp"
#include "optimization/gaussian/gpr.hpp"
#include "optimization/gaussian/kernel/matern_52.hpp"
#include "processing/image/comparator.hpp"

template<FloatingPoint T = double>
class ViewpointEvaluator {
public:
    ViewpointEvaluator(optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
                       cache::ViewpointCache<T> &cache,
                       optimization::Acquisition<T> &acquisition, const int patience,
                       T improvement_threshold)
        : gpr_(gpr), cache_(cache), acquisition_(acquisition),
          patience_(patience), improvement_threshold_(improvement_threshold) {}

    void evaluatePoint(ViewPoint<T> &point, const Image<> &target,
                       const std::shared_ptr<processing::image::ImageComparator> &comparator) {
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

            gpr_.update(point.getPosition(), point.getScore());
        } else {
            // Update the cache with the existing point
            cache_.update(point);
        }
    }

    T computeAcquisition(const Eigen::Vector3<T> &x, T mean, T std_dev) const {
        acquisition_.incrementIteration();
        return acquisition_.compute(x, mean, std_dev);
    }

    bool hasConverged(T current_score, T best_score, T target_score, int current_iteration,
                      std::deque<T> &recent_scores, int &stagnant_iterations, const ViewPoint<T>& best_viewpoint) {
        // Check if we've reached or exceeded the target score
        if (current_score >= target_score) {
            LOG_INFO("Target score reached at iteration {}", current_iteration);
            return true;
        }

        LOG_INFO("Current score: {}, Best score: {}", current_score, best_score);

        // Calculate relative improvement
        T relative_improvement = (current_score - best_score) / best_score;

        if (relative_improvement > improvement_threshold_) {
            stagnant_iterations = 0;
        } else {
            stagnant_iterations++;
        }

        // Early stopping based on stagnation
        if (stagnant_iterations >= patience_) {
            LOG_INFO("Early stopping triggered after {} stagnant iterations", patience_);
            return true;
        }

        // Moving average convergence check
        recent_scores.push_back(current_score);
        if (recent_scores.size() > window_size_) {
            recent_scores.pop_front();
        }

        if (recent_scores.size() == window_size_) {
            T avg_score = std::reduce(recent_scores.begin(), recent_scores.end(), T(0)) / window_size_;
            T score_variance =
                    std::accumulate(recent_scores.begin(), recent_scores.end(), T(0),
                                    [avg_score](T acc, T score) { return acc + std::pow(score - avg_score, 2); }) /
                    window_size_;

            if (score_variance < T(1e-6) && avg_score > target_score * T(0.95)) {
                LOG_INFO("Convergence detected based on moving average at iteration {}", current_iteration);
                return true;
            }
        }

        // Check confidence interval using GPR
        auto [mean, variance] = gpr_.predict(best_viewpoint.getPosition());
        T confidence_interval = T(1.96) * std::sqrt(variance); // 95% confidence interval

        if (mean - confidence_interval > target_score) {
            LOG_INFO("High confidence in solution at iteration {}", current_iteration);
            return true;
        }

        return false;
    }

private:
    optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr_;
    cache::ViewpointCache<T> &cache_;
    optimization::Acquisition<T> &acquisition_;
    int patience_;
    T improvement_threshold_;
    static constexpr int window_size_ = 5;
};

#endif //VIEWPOINT_EVALUATOR_HPP
