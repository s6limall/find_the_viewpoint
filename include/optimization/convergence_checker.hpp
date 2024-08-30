// File: optimization/convergence_checker.hpp

#ifndef CONVERGENCE_CHECKER_HPP
#define CONVERGENCE_CHECKER_HPP

#include <cmath>
#include <deque>
#include <numeric>
#include <optional>
#include "types/viewpoint.hpp"

namespace optimization {

    template<typename T>
    class ConvergenceChecker {
    public:
        struct Config {
            int min_iterations; // Minimum number of iterations before checking convergence
            int max_iterations; // Maximum number of iterations
            T absolute_tolerance; // Absolute tolerance for improvement
            T relative_tolerance; // Relative tolerance for improvement
            int patience; // Number of iterations to wait for improvement
            T target_score; // Target score to reach
            int window_size; // Window size for moving average calculations
        };

        explicit ConvergenceChecker(const Config &config) :
            config_(config), iterations_(0), iterations_since_improvement_(0),
            best_score_(std::numeric_limits<T>::lowest()) {}

        bool hasConverged(T current_score, const ViewPoint<T> &current_viewpoint) {
            ++iterations_;
            updateScoreHistory(current_score);

            if (iterations_ < config_.min_iterations) {
                return false;
            }

            if (iterations_ >= config_.max_iterations) {
                LOG_INFO("Maximum iterations reached. Declaring convergence.");
                return true;
            }

            if (current_score >= config_.target_score) {
                LOG_INFO("Target score reached. Declaring convergence.");
                return true;
            }

            if (current_score > best_score_) {
                best_score_ = current_score;
                best_viewpoint_ = current_viewpoint;
                iterations_since_improvement_ = 0;
            } else {
                ++iterations_since_improvement_;
            }

            if (iterations_since_improvement_ >= config_.patience) {
                LOG_INFO("No improvement for {} iterations. Declaring convergence.", config_.patience);
                return true;
            }

            if (score_history_.size() >= config_.window_size) {
                if (hasAbsoluteConvergence() && hasRelativeConvergence()) {
                    LOG_INFO("Absolute and relative convergence criteria met. Declaring convergence.");
                    return true;
                }
            }

            return false;
        }

        void reset() {
            iterations_ = 0;
            iterations_since_improvement_ = 0;
            best_score_ = std::numeric_limits<T>::lowest();
            best_viewpoint_ = std::nullopt;
            score_history_.clear();
        }

        std::optional<ViewPoint<T>> getBestViewpoint() const { return best_viewpoint_; }

    private:
        Config config_;
        int iterations_;
        int iterations_since_improvement_;
        T best_score_;
        std::optional<ViewPoint<T>> best_viewpoint_;
        std::deque<T> score_history_;

        void updateScoreHistory(T score) {
            score_history_.push_back(score);
            if (score_history_.size() > config_.window_size) {
                score_history_.pop_front();
            }
        }

        [[nodiscard]] bool hasAbsoluteConvergence() const {
            T improvement = score_history_.back() - score_history_.front();
            return std::abs(improvement) < config_.absolute_tolerance;
        }

        [[nodiscard]] bool hasRelativeConvergence() const {
            T relative_improvement =
                    (score_history_.back() - score_history_.front()) / std::abs(score_history_.front());
            return std::abs(relative_improvement) < config_.relative_tolerance;
        }
    };

} // namespace optimization

#endif // CONVERGENCE_CHECKER_HPP
