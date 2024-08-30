// File: optimization/multistart_optimizer.hpp

#ifndef MULTISTART_OPTIMIZER_HPP
#define MULTISTART_OPTIMIZER_HPP

// File: optimization/modern_multi_start_optimizer.hpp

#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <random>
#include <algorithm>
#include <execution>
#include <Eigen/Dense>

#include "viewpoint_optimizer.hpp"
#include "gaussian/gpr.hpp"
#include "gaussian/kernel/matern_52.hpp"
#include "acquisition.hpp"
#include "cache/viewpoint_cache.hpp"
#include "evaluation/viewpoint_evaluator.hpp"
#include "sampling/viewpoint_sampler.hpp"
#include "spatial/octree.hpp"
#include "common/logging/logger.hpp"
#include "config/configuration.hpp"

namespace optimization {

template<typename T>
class MultiStartViewpointOptimizer {
public:
    struct Config {
        Eigen::Vector3<T> center;
        T size;
        T min_size;
        int max_iterations;
        int max_restarts;
        T improvement_threshold;
        T target_score;
    };

    explicit MultiStartViewpointOptimizer(const Config& config)
        : config_(config), rng_(std::random_device{}()) {}

    [[nodiscard]] std::optional<ViewPoint<T>> optimize(const Image<>& target,
                                                       const std::shared_ptr<processing::image::ImageComparator>& comparator) {
        std::optional<ViewPoint<T>> global_best_viewpoint;
        T global_best_score = std::numeric_limits<T>::lowest();

        for (int restart = 0; restart < config_.max_restarts; ++restart) {
            LOG_INFO("Starting optimization attempt {} of {}", restart + 1, config_.max_restarts);

            try {
                auto [best_viewpoint, best_score] = runSingleOptimization(target, comparator);

                if (best_viewpoint && best_score > global_best_score) {
                    global_best_viewpoint = best_viewpoint;
                    global_best_score = best_score;
                    LOG_INFO("New global best viewpoint found: {}, Score: {}",
                             best_viewpoint->toString(), best_score);
                }

                if (global_best_score >= config_.target_score) {
                    LOG_INFO("Target score reached. Stopping optimization.");
                    break;
                }

                if (restart < config_.max_restarts - 1) {
                    LOG_INFO("Preparing for next optimization attempt.");
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Error during optimization attempt {}: {}. Continuing with next attempt.",
                          restart + 1, e.what());
            }
        }

        if (global_best_viewpoint) {
            LOG_INFO("Multi-start optimization complete. Best viewpoint: {}, Score: {}",
                     global_best_viewpoint->toString(), global_best_score);
        } else {
            LOG_WARN("No suitable viewpoint found after {} restart(s)", config_.max_restarts);
        }

        return global_best_viewpoint;
    }

private:
    Config config_;
    std::mt19937 rng_;

    [[nodiscard]] std::pair<std::optional<ViewPoint<T>>, T> runSingleOptimization(
        const Image<>& target,
        const std::shared_ptr<processing::image::ImageComparator>& comparator) {

        auto kernel = std::make_unique<optimization::kernel::Matern52<T>>();
        auto gpr = std::make_unique<optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>>>(*kernel);

        auto cache = std::make_unique<cache::ViewpointCache<T>>(typename cache::ViewpointCache<T>::CacheConfig{});
        auto acquisition = std::make_unique<optimization::Acquisition<T>>(typename optimization::Acquisition<T>::Config{});
        auto sampler = std::make_unique<ViewpointSampler<T>>(config_.center, config_.size / 2, 0.1);
        auto evaluator = std::make_unique<ViewpointEvaluator<T>>(*gpr, *cache, *acquisition, 10, config_.improvement_threshold);

        auto optimizer = std::make_unique<ViewpointOptimizer<T>>(
            config_.center, config_.size, config_.min_size, config_.max_iterations, *gpr,
            config_.size / 2, 0.1
        );

        auto initial_samples = sampler->samplePoints(typename spatial::Octree<T>::Node(config_.center, config_.size));
        auto initial_best = selectInitialBest(initial_samples, target, comparator);

        optimizer->optimize(target, comparator, initial_best, config_.target_score);

        return {optimizer->getBestViewpoint(), optimizer->getBestViewpoint() ?
                optimizer->getBestViewpoint()->getScore() : std::numeric_limits<T>::lowest()};
    }

    [[nodiscard]] ViewPoint<T> selectInitialBest(const std::vector<ViewPoint<T>>& samples,
                                                 const Image<>& target,
                                                 const std::shared_ptr<processing::image::ImageComparator>& comparator) const {
        auto best_sample = std::max_element(samples.begin(), samples.end(),
            [&](const auto& a, const auto& b) {
                return comparator->compare(target, Image<>::fromViewPoint(a)) <
                       comparator->compare(target, Image<>::fromViewPoint(b));
            });

        T best_score = comparator->compare(target, Image<>::fromViewPoint(*best_sample));
        ViewPoint<T> best_viewpoint = *best_sample;
        best_viewpoint.setScore(best_score);

        LOG_INFO("Selected initial best viewpoint: {}, Score: {}",
                 best_viewpoint.toString(), best_score);
        return best_viewpoint;
    }
};

} // namespace optimization

#endif //MULTISTART_OPTIMIZER_HPP
