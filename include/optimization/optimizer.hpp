// File: optimization/optimizer.hpp

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>
#include "core/view.hpp"
#include "common/logging/logger.hpp"

namespace optimization {

    struct OptimizationResult {
        std::vector<core::View> optimized_views;
        double best_score;
        int iterations;
        bool success;
    };

    /**
     * @brief Abstract base class for optimization algorithms.
     */
    class Optimizer {
    public:
        virtual ~Optimizer() = default;

        /**
         * @brief Perform optimization to find the best viewpoint matching the target image.
         * @param initial_views Initial viewpoints to start the optimization.
         * @param target_image The image we want to match.
         * @param evaluate_callback Callback function to evaluate the score of each view.
         * @return The result of the optimization.
         */
        virtual OptimizationResult optimize(const std::vector<core::View> &initial_views,
                                            const cv::Mat &target_image,
                                            std::function<double(const core::View &, const cv::Mat &)>
                                            evaluate_callback) = 0;

        /**
         * @brief Update the optimizer with feedback from the evaluation.
         * @param view The view being evaluated.
         * @param score The score of the view.
         * @return Whether the update was successful.
         */
        virtual bool update(const core::View &view, double score) = 0;

        /**
         * @brief Get the next viewpoint to be evaluated.
         * @return The next viewpoint.
         */
        virtual core::View getNextView() = 0;

        /**
         * @brief Get cached evaluation score for a view.
         * @param view The view to check the cache.
         * @return Cached score if available, else NaN.
         */
        [[nodiscard]] virtual double getCachedScore(const core::View &view) const = 0;

        /**
         * @brief Set cache for a view's score.
         * @param view The view.
         * @param score The score.
         */
        virtual void setCache(const core::View &view, double score) = 0;

    protected:
        std::shared_ptr<spdlog::logger> logger = spdlog::default_logger();
    };

}

#endif // OPTIMIZER_HPP
