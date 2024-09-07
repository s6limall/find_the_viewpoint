// File: optimization/convergence_checker.hpp

#ifndef CONVERGENCE_CHECKER_HPP
#define CONVERGENCE_CHECKER_HPP

#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/gpr.hpp"
#include "optimization/gaussian/kernel/matern_52.hpp"
#include "types/viewpoint.hpp"

namespace optimization {

    template<FloatingPoint T = double>
    class ConvergenceChecker {
    public:
        ConvergenceChecker(const int initial_patience, T initial_improvement_threshold, T target_score,
                           const int window_size = 5) :
            initial_patience_(initial_patience), patience_(initial_patience),
            initial_improvement_threshold_(initial_improvement_threshold),
            improvement_threshold_(initial_improvement_threshold), target_score_(target_score),
            window_size_(window_size), stagnant_iterations_(0), total_iterations_(0) {}

        bool hasConverged(T current_score, T best_score, int current_iteration, const ViewPoint<T> &best_viewpoint,
                          const GPR<kernel::Matern52<T>> &gpr) {
            total_iterations_++;
            updateAdaptiveParameters();

            if (current_score >= target_score_) {
                LOG_INFO("Target score reached at iteration {}", current_iteration);
                return true;
            }

            LOG_INFO("Current score: {}, Best score: {}", current_score, best_score);

            // Calculate relative improvement
            T relative_improvement = (current_score - best_score) / best_score;

            if (relative_improvement > improvement_threshold_) {
                stagnant_iterations_ = 0;
            } else {
                stagnant_iterations_++;
            }

            // Early stopping based on stagnation
            if (stagnant_iterations_ >= patience_) {
                LOG_INFO("Early stopping triggered after {} stagnant iterations", patience_);
                return true;
            }

            // Adaptive moving average convergence check
            recent_scores_.push_back(current_score);
            if (recent_scores_.size() > window_size_) {
                recent_scores_.pop_front();
            }

            if (recent_scores_.size() == window_size_) {
                T avg_score = std::reduce(recent_scores_.begin(), recent_scores_.end(), T(0)) / window_size_;
                T score_variance = calculateScoreVariance(avg_score);

                T adaptive_threshold = std::max(T(1e-6), T(1e-4) * std::pow(0.95, total_iterations_));
                if (score_variance < adaptive_threshold && avg_score > target_score_ * T(0.98)) {
                    LOG_INFO("Convergence detected based on adaptive moving average at iteration {}",
                             current_iteration);
                    return true;
                }
            }

            // Check confidence interval using GPR
            auto [mean, variance] = gpr.predict(best_viewpoint.getPosition());
            T confidence_interval = T(1.96) * std::sqrt(variance); // 95% confidence interval

            // Adaptive confidence threshold
            T adaptive_confidence_threshold =
                    target_score_ * (T(1) - T(0.02) * std::exp(-T(total_iterations_) / T(20)));
            if (mean - confidence_interval > adaptive_confidence_threshold) {
                LOG_INFO("High confidence in solution at iteration {}", current_iteration);
                return true;
            }

            // Check for rapid improvement
            if (relative_improvement > T(0.1)) {
                LOG_INFO("Significant improvement detected. Continuing optimization.");
                return false;
            }

            // Check for diminishing returns
            if (total_iterations_ > 10 && calculateAverageImprovement() < improvement_threshold_ / 10) {
                LOG_INFO("Diminishing returns detected. Stopping optimization.");
                return true;
            }

            return false;
        }

        void reset() {
            stagnant_iterations_ = 0;
            total_iterations_ = 0;
            recent_scores_.clear();
            improvement_history_.clear();
            patience_ = initial_patience_;
            improvement_threshold_ = initial_improvement_threshold_;
        }

        void setTargetScore(T target_score) { target_score_ = target_score; }

    private:
        int initial_patience_;
        int patience_;
        T initial_improvement_threshold_;
        T improvement_threshold_;
        T target_score_;
        size_t window_size_;
        int stagnant_iterations_;
        int total_iterations_;
        std::deque<T> recent_scores_;
        std::deque<T> improvement_history_;

        void updateAdaptiveParameters() {
            // Adaptively adjust patience
            patience_ = initial_patience_ + static_cast<int>(std::log1p(total_iterations_));

            // Adaptively adjust improvement threshold
            improvement_threshold_ = initial_improvement_threshold_ * std::exp(-T(total_iterations_) / T(50));
        }

        T calculateScoreVariance(T avg_score) const {
            return std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0),
                                   [avg_score](T acc, T score) { return acc + std::pow(score - avg_score, 2); }) /
                   window_size_;
        }

        T calculateAverageImprovement() const {
            if (improvement_history_.size() < 2)
                return T(0);

            T sum_improvement = T(0);
            for (size_t i = 1; i < improvement_history_.size(); ++i) {
                sum_improvement +=
                        (improvement_history_[i] - improvement_history_[i - 1]) / improvement_history_[i - 1];
            }
            return sum_improvement / T(improvement_history_.size() - 1);
        }
    };

} // namespace optimization

#endif // CONVERGENCE_CHECKER_HPP
