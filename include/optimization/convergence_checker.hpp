// File: optimization/convergence_checker.hpp

#ifndef CONVERGENCE_CHECKER_HPP
#define CONVERGENCE_CHECKER_HPP

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <optional>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/gpr.hpp"
#include "types/viewpoint.hpp"

namespace optimization {

    template<FloatingPoint T = double, IsKernel<T> KernelType = DefaultKernel<T>>
    class ConvergenceChecker {
    public:
        struct Config {
            T improvement_threshold = 1e-4;
            int patience = 10;
            T target_score = 0.95;
            int window_size = 5;
            T confidence_threshold = 0.9;
            T variance_factor = 0.02;
            bool rapid_improvement_check = true;
            bool diminishing_returns_check = true;
        };

        explicit ConvergenceChecker(Config config = Config()) :
            config_(std::move(config)), patience_(config_.patience),
            improvement_threshold_(config_.improvement_threshold) {
            // Assign configuration values
            config_.patience = config::get("optimization.patience", config_.patience);
            config_.improvement_threshold =
                    config::get("optimization.improvement_threshold", config_.improvement_threshold);
            /*config_.target_score =
                    state::get("target_score", config::get("optimization.target_score", config_.target_score));*/
            config_.target_score = state::get("target_score", config_.target_score);
            config_.window_size = config::get("optimization.window_size", config_.window_size);
            config_.confidence_threshold =
                    config::get("optimization.confidence_threshold", config_.confidence_threshold);
            config_.variance_factor = config::get("optimization.variance_factor", config_.variance_factor);
            config_.rapid_improvement_check =
                    config::get("optimization.rapid_improvement_check", config_.rapid_improvement_check);
            config_.diminishing_returns_check =
                    config::get("optimization.diminishing_returns_check", config_.diminishing_returns_check);
        }

        bool hasConverged(T current_score, T best_score, const int iteration, const ViewPoint<T> &best_viewpoint,
                          std::shared_ptr<GPR<T, KernelType>> gpr) {
            total_iterations_++;
            updateAdaptiveParameters();

            if (current_score >= config_.target_score)
                return logAndReturn(true, "Target score reached", iteration);

            T rel_improvement = (current_score - best_score) / std::max(best_score, T(1e-8));
            stagnant_iterations_ = (rel_improvement > improvement_threshold_) ? 0 : stagnant_iterations_ + 1;

            if (stagnant_iterations_ >= patience_)
                return logAndReturn(true, "Early stopping due to stagnation", iteration);
            if (checkMovingAverageConvergence(current_score))
                return logAndReturn(true, "Moving average convergence", iteration);
            if (config_.confidence_threshold > 0 && checkConfidenceInterval(best_viewpoint, gpr))
                return logAndReturn(true, "High confidence in solution", iteration);
            if (config_.rapid_improvement_check && rel_improvement > 0.1)
                return false;
            if (config_.diminishing_returns_check && checkDiminishingReturns())
                return logAndReturn(true, "Diminishing returns", iteration);

            return false;
        }

        void reset() {
            stagnant_iterations_ = 0;
            total_iterations_ = 0;
            recent_scores_.clear();
            improvement_history_.clear();
            patience_ = config_.patience;
            improvement_threshold_ = config_.improvement_threshold;
        }

        void setTargetScore(T target_score) { config_.target_score = target_score; }
        void setConfidenceThreshold(T confidence_threshold) { config_.confidence_threshold = confidence_threshold; }

    private:
        Config config_;
        int patience_, stagnant_iterations_ = 0, total_iterations_ = 0;
        T improvement_threshold_;
        std::deque<T> recent_scores_, improvement_history_;

        void updateAdaptiveParameters() {
            patience_ = config_.patience + static_cast<int>(std::log1p(total_iterations_));
            improvement_threshold_ = config_.improvement_threshold * std::exp(-T(total_iterations_) / T(50));
        }

        bool checkMovingAverageConvergence(T current_score) {
            recent_scores_.push_back(current_score);
            if (recent_scores_.size() > config_.window_size)
                recent_scores_.pop_front();

            if (recent_scores_.size() == config_.window_size) {
                T avg_score = std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0)) / config_.window_size;
                T variance =
                        std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0),
                                        [avg_score](T acc, T score) { return acc + std::pow(score - avg_score, 2); }) /
                        config_.window_size;

                T adaptive_threshold = std::max(T(1e-6), T(1e-4) * std::pow(0.95, total_iterations_));
                return variance < adaptive_threshold && avg_score > config_.target_score * T(0.98);
            }
            return false;
        }

        bool checkConfidenceInterval(const ViewPoint<T> &best_viewpoint, std::shared_ptr<GPR<T, KernelType>> gpr) {
            auto [mean, variance] = gpr->predict(best_viewpoint.getPosition());
            T confidence_interval = T(1.96) * std::sqrt(variance);

            T adaptive_confidence_threshold =
                    config_.target_score * (T(1) - config_.variance_factor * std::exp(-T(total_iterations_) / T(20)));

            bool is_high_confidence =
                    mean - confidence_interval > adaptive_confidence_threshold * config_.confidence_threshold;
            return is_high_confidence && mean > config_.target_score;
        }

        [[nodiscard]] bool checkDiminishingReturns() const {
            if (improvement_history_.size() < 2)
                return false;

            T avg_improvement =
                    std::accumulate(improvement_history_.begin() + 1, improvement_history_.end(), T(0),
                                    [this](T sum, T score) {
                                        return sum + (score - improvement_history_[0]) / std::max(score, T(1e-8));
                                    }) /
                    T(improvement_history_.size() - 1);

            return total_iterations_ > 10 && avg_improvement < improvement_threshold_ / 10;
        }

        static bool logAndReturn(const bool result, const std::string &message, int iteration) {
            LOG_INFO("ConvergenceChecker: {} at iteration {}", message, iteration);
            return result;
        }
    };

} // namespace optimization

#endif // CONVERGENCE_CHECKER_HPP
