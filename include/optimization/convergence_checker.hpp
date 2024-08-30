// File: optimization/convergence_checker.hpp

#ifndef CONVERGENCE_CHECKER_HPP
#define CONVERGENCE_CHECKER_HPP

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <vector>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/gpr.hpp"
#include "types/viewpoint.hpp"

namespace optimization {

    template<FloatingPoint T = double>
    class ConvergenceChecker {
    public:
        struct Config {
            T target_score;
            int max_iterations;
            int patience;
            T improvement_threshold;
            int window_size;
            T moving_average_threshold;
            T confidence_interval_threshold;
            T adaptive_threshold_factor;
            std::vector<T> criteria_weights;
        };

        ConvergenceChecker(const Config &config,
                           optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr) :
            config_(config), gpr_(gpr), stagnant_iterations_(0), recent_scores_(), iteration_(0) {
            if (config_.criteria_weights.empty()) {
                config_.criteria_weights = {1.0, 1.0, 1.0, 1.0, 1.0}; // Equal weights by default
            }
            normalizeWeights();
        }

        bool hasConverged(const ViewPoint<T> &best_viewpoint, T current_score, T best_score) {
            iteration_++;
            updateRecentScores(current_score);

            std::vector<bool> convergence_votes;
            std::vector<T> vote_strengths;

            // 1. Target score achievement
            const bool target_achieved = checkTargetScore(current_score);
            convergence_votes.push_back(target_achieved);
            vote_strengths.push_back(target_achieved ? 1.0 : 0.0);

            // 2. Improvement rate
            const bool significant_improvement = checkImprovementRate(current_score, best_score);
            convergence_votes.push_back(!significant_improvement);
            vote_strengths.push_back(significant_improvement ? 0.0 : 1.0);

            // 3. Stagnation detection
            const bool stagnated = checkStagnation();
            convergence_votes.push_back(stagnated);
            vote_strengths.push_back(stagnated ? 1.0 : 0.0);

            // 4. Moving average analysis
            const bool moving_average_converged = checkMovingAverage();
            convergence_votes.push_back(moving_average_converged);
            vote_strengths.push_back(moving_average_converged ? 1.0 : 0.0);

            // 5. Gaussian Process-based uncertainty estimation
            const bool high_confidence = checkConfidenceInterval(best_viewpoint);
            convergence_votes.push_back(high_confidence);
            vote_strengths.push_back(high_confidence ? 1.0 : 0.0);

            // Compute weighted vote
            T weighted_vote = std::inner_product(vote_strengths.begin(), vote_strengths.end(),
                                                 config_.criteria_weights.begin(), T(0));
            T vote_threshold = adaptVoteThreshold();

            bool converged = weighted_vote >= vote_threshold;

            LOG_INFO("Convergence check at iteration {}: weighted vote = {}, threshold = {}, converged = {}",
                     iteration_, weighted_vote, vote_threshold, converged);

            return converged || iteration_ >= config_.max_iterations;
        }

    private:
        Config config_;
        optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr_;
        int stagnant_iterations_;
        std::deque<T> recent_scores_;
        int iteration_;

        void normalizeWeights() {
            T sum = std::accumulate(config_.criteria_weights.begin(), config_.criteria_weights.end(), T(0));
            std::transform(config_.criteria_weights.begin(), config_.criteria_weights.end(),
                           config_.criteria_weights.begin(), [sum](T w) { return w / sum; });
        }

        void updateRecentScores(T current_score) {
            recent_scores_.push_back(current_score);
            if (recent_scores_.size() > config_.window_size) {
                recent_scores_.pop_front();
            }
        }

        bool checkTargetScore(T current_score) const { return current_score >= config_.target_score; }

        bool checkImprovementRate(T current_score, T best_score) {
            T relative_improvement = (current_score - best_score) / std::max(std::abs(best_score), T(1e-10));
            const bool significant_improvement = relative_improvement > config_.improvement_threshold;

            if (significant_improvement) {
                stagnant_iterations_ = 0;
            } else {
                stagnant_iterations_++;
            }

            return significant_improvement;
        }

        [[nodiscard]] bool checkStagnation() const { return stagnant_iterations_ >= config_.patience; }

        [[nodiscard]] bool checkMovingAverage() const {
            if (recent_scores_.size() < config_.window_size) {
                return false;
            }

            T avg_score = std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0)) / config_.window_size;
            T score_variance =
                    std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0),
                                    [avg_score](T acc, T score) { return acc + std::pow(score - avg_score, 2); }) /
                    config_.window_size;

            return score_variance < config_.moving_average_threshold && avg_score > config_.target_score * T(0.95);
        }

        bool checkConfidenceInterval(const ViewPoint<T> &best_viewpoint) const {
            auto [mean, variance] = gpr_.predict(best_viewpoint.getPosition());
            T confidence_interval = T(1.96) * std::sqrt(variance); // 95% confidence interval
            return mean - confidence_interval > config_.target_score * config_.confidence_interval_threshold;
        }

        T adaptVoteThreshold() const {
            T progress_ratio = static_cast<T>(iteration_) / config_.max_iterations;
            T adaptive_factor = std::exp(-config_.adaptive_threshold_factor * progress_ratio);
            return T(0.5) + T(0.5) * adaptive_factor; // Starts at 1.0 and decreases towards 0.5
        }
    };

} // namespace optimization

#endif // CONVERGENCE_CHECKER_HPP
