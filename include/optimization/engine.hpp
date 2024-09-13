// File: optimization/engine.hpp

#ifndef OPTIMIZATION_ENGINE_HPP
#define OPTIMIZATION_ENGINE_HPP

#include "cache/viewpoint_cache.hpp"
#include "common/logging/logger.hpp"
#include "common/metrics/metrics_collector.hpp"
#include "evaluation/viewpoint_evaluator.hpp"
#include "optimization/acquisition.hpp"
#include "processing/image/comparator.hpp"
#include "sampling/viewpoint_sampler.hpp"
#include "spatial/octree.hpp"

namespace optimization {

    template<FloatingPoint T = double, IsKernel<T> KernelType = DefaultKernel<T>>
    class OptimizationEngine {
    public:
        OptimizationEngine(const Eigen::Vector3<T> &center, T size, T min_size, std::shared_ptr<GPR<T, KernelType>> gpr,
                           std::optional<T> radius = std::nullopt, std::optional<T> tolerance = std::nullopt) :
            octree_(center, size, min_size), cache_(typename cache::ViewpointCache<T>::CacheConfig{}), gpr_(gpr),
            acquisition_(typename Acquisition<T>::Config{}),
            sampler_(center, radius.value_or(0), tolerance.value_or(0), gpr_, acquisition_),
            convergence_checker_(typename ConvergenceChecker<T, KernelType>::Config{}),
            evaluator_(gpr, cache_, acquisition_, convergence_checker_), min_size_(min_size),
            max_points_(config::get("optimization.max_points", 0)),
            max_iterations_(config::get("optimization.max_iterations", 5)) {}


        std::optional<ViewPoint<T>> refine(const Image<> &target,
                                           const std::shared_ptr<processing::image::ImageComparator> &comparator,
                                           int current_iteration) {
            if (hasReachedMaxPoints()) {
                LOG_WARN("Max points reached. Stopping optimization, assuming refined.");
                return best_viewpoint_;
            }

            LOG_DEBUG("Starting optimization - refinement iteration {}", current_iteration);
            LOG_DEBUG("Updating acquisition function");
            updateAcquisitionFunction(current_iteration);

            std::priority_queue<NodeScore> pq;
            const T distance_scale = octree_.getRoot()->size / 10.0;
            pq.emplace(octree_.getRoot()->max_acquisition, 0.0, octree_.getRoot());

            int nodes_explored = 0;
            const int min_nodes_to_explore = config::get<int>("octree.min_nodes_to_explore", 5);
            T best_score_this_refinement =
                    best_viewpoint_ ? best_viewpoint_->getScore() : std::numeric_limits<T>::lowest();

            while (!pq.empty() && (nodes_explored < min_nodes_to_explore || current_iteration < max_iterations_ / 2)) {
                if (hasReachedMaxPoints()) {
                    LOG_WARN("Max points reached during iteration. Stopping optimization, assuming refined.");
                    return best_viewpoint_;
                }

                auto [acquisition_value, distance_penalty, node] = pq.top();
                pq.pop();

                if (node->size < min_size_)
                    continue;

                exploreNode(*node, target, comparator);
                nodes_explored++;

                T current_best_score = best_viewpoint_ ? best_viewpoint_->getScore() : std::numeric_limits<T>::lowest();
                if (current_best_score > best_score_this_refinement) {
                    best_score_this_refinement = current_best_score;
                    LOG_INFO("Best score updated to {} at iteration {} - {} nodes explored.",
                             best_score_this_refinement, current_iteration, nodes_explored);

                    focusExplorationAroundBest(pq);
                }

                if (!node->isLeaf()) {
                    for (const auto &child: node->children) {
                        if (child) {
                            T dist_to_best =
                                    best_viewpoint_ ? (child->center - best_viewpoint_->getPosition()).norm() : 0;
                            T child_distance_penalty = std::exp(-dist_to_best / distance_scale);
                            T child_acquisition = child->max_acquisition * std::exp(-dist_to_best / (node->size / 2));
                            pq.emplace(child_acquisition, child_distance_penalty, child.get());
                        }
                    }
                }

                // Early stopping condition
                if (current_best_score > config::get<T>("optimization.target_score", 0.9)) {
                    LOG_INFO("Target score reached. Stopping optimization.");
                    break;
                }
            }

            return best_viewpoint_;
        }

        void exploreNode(typename spatial::Octree<T>::Node &node, const Image<> &target,
                         const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            if (node.points.empty()) {
                LOG_WARN("Node has no points. Attempting to sample points.");
                node.points = sampler_.samplePoints(node, best_viewpoint_ ? &(*best_viewpoint_) : nullptr);

                if (node.points.empty()) {
                    LOG_ERROR("Failed to sample any valid points for node.");
                    return; //  skip node
                }
            }

            LOG_DEBUG("Exploring node with center {} and size {}", node.center, node.size);


            LOG_DEBUG("Exploring node with center {} and size {}", node.center, node.size);

            node.max_acquisition = std::numeric_limits<T>::lowest();
            for (auto &point: node.points) {
                evaluator_.evaluatePoint(point, target, comparator);
                auto [mean, std_dev] = gpr_->predict(point.getPosition());
                T acquisition_value = acquisition_.compute(point.getPosition(), mean, std_dev);
                node.max_acquisition = std::max(node.max_acquisition, acquisition_value);

                metrics::recordMetrics(point, {{"position_x", point.getPosition().x()},
                                               {"position_y", point.getPosition().y()},
                                               {"position_z", point.getPosition().z()},
                                               {"score", point.getScore()},
                                               {"acquisition_value", acquisition_value}});

                updateBestViewpoint(point);
            }

            if (octree_.shouldSplit(node, best_viewpoint_)) {
                octree_.split(node);
                // Add points around best viewpoint in new child nodes if applicable
                for (const auto &child: node.children) {
                    if (child && best_viewpoint_ && octree_.isWithinNode(*child, best_viewpoint_->getPosition())) {
                        sampler_.addPointsAroundBest(*child, *best_viewpoint_);
                    }
                }
            }

            node.explored = true;
        }

        void updateBestViewpoint(const ViewPoint<T> &point) {
            if (!best_viewpoint_ || point.getScore() > best_viewpoint_->getScore()) {
                best_viewpoint_ = point;
                LOG_INFO("New best viewpoint: {} with score {}", point.toString(), point.getScore());
            }
        }

        void updateAcquisitionFunction(const int current_iteration) {
            T recent_improvement_rate = calculateRecentImprovementRate();
            T current_best_score = best_viewpoint_ ? best_viewpoint_->getScore() : 0;

            auto current_config = acquisition_.getConfig();

            const T low_score_threshold = 0.3;
            const T high_score_threshold = 0.7;

            T base_exploration_weight = std::max(0.1, 1.0 - current_iteration / static_cast<double>(max_iterations_));

            if (current_best_score < low_score_threshold || recent_improvement_rate > 0.05) {
                current_config.exploration_weight = std::min(1.0, base_exploration_weight * 1.5);
            } else if (current_best_score > high_score_threshold) {
                current_config.exploration_weight = std::max(0.1, base_exploration_weight * 0.5);
            } else {
                current_config.exploration_weight = base_exploration_weight;
            }

            current_config.exploitation_weight = 1.0 - current_config.exploration_weight;
            current_config.momentum = std::max(0.05, std::min(0.5, recent_improvement_rate * 5));

            acquisition_.updateConfig(current_config);

            LOG_DEBUG("Updated acquisition function: exploration_weight={}, exploitation_weight={}, momentum={}",
                      current_config.exploration_weight, current_config.exploitation_weight, current_config.momentum);
        }

        T calculateRecentImprovementRate() const {
            const int window_size = std::min(10, static_cast<int>(recent_scores_.size()));
            if (window_size < 2)
                return 0.0; // Not enough data

            T oldest_score = recent_scores_[recent_scores_.size() - window_size];
            T newest_score = recent_scores_.back();

            return (newest_score - oldest_score) / static_cast<T>(window_size - 1);
        }

        [[nodiscard]] bool hasReachedMaxPoints() const noexcept {
            return state::get("count", 0) > static_cast<int>(max_points_) && max_points_ > 0;
        }

        void setMaxIterations(const int max_iterations) { max_iterations_ = max_iterations; }

        std::optional<ViewPoint<T>> getBestViewpoint() const { return best_viewpoint_; }

        void addRecentScore(T score) {
            recent_scores_.push_back(score);
            if (recent_scores_.size() > 10) {
                recent_scores_.pop_front();
            }
        }

        void updateExplorationRate(T stagnation_ratio) {
            T exploration_rate = std::max(T(0.1), T(1.0) - stagnation_ratio);
            T exploitation_rate = T(1.0) - exploration_rate;

            typename Acquisition<T>::Config new_config = acquisition_.getConfig();
            new_config.exploration_weight = exploration_rate;
            new_config.exploitation_weight = exploitation_rate;

            acquisition_.updateConfig(new_config);

            LOG_INFO("Updated exploration rate to {}, exploitation rate to {}", exploration_rate, exploitation_rate);
        }

    private:
        spatial::Octree<T> octree_;
        cache::ViewpointCache<T> cache_;
        std::shared_ptr<GPR<T, KernelType>> gpr_;
        Acquisition<T> acquisition_;
        ViewpointSampler<T> sampler_;
        ConvergenceChecker<T, KernelType> convergence_checker_;
        ViewpointEvaluator<T> evaluator_;

        T min_size_;
        size_t max_points_;
        int max_iterations_;
        std::optional<ViewPoint<T>> best_viewpoint_;
        std::deque<T> recent_scores_;

        struct NodeScore {
            T acquisition_value;
            T distance_penalty;
            typename spatial::Octree<T>::Node *node;

            NodeScore(T acquisition, T dist_penalty, typename spatial::Octree<T>::Node *n) :
                acquisition_value(acquisition), distance_penalty(dist_penalty), node(n) {}

            bool operator<(const NodeScore &other) const {
                return (acquisition_value - distance_penalty) < (other.acquisition_value - other.distance_penalty);
            }
        };


        void focusExplorationAroundBest(std::priority_queue<NodeScore> &pq) {
            if (!best_viewpoint_)
                return;

            std::priority_queue<NodeScore> new_pq;
            const T focus_radius = octree_.getRoot()->size / 4;
            const T focus_boost = 2.0;
            const T penalization_factor = 0.5;

            while (!pq.empty()) {
                auto [acquisition_value, distance_penalty, node] = pq.top();
                pq.pop();

                T dist_to_best = (node->center - best_viewpoint_->getPosition()).norm();
                T scaled_dist = dist_to_best / focus_radius;

                if (scaled_dist < 1) {
                    // Boost nodes close to the best viewpoint
                    new_pq.emplace(acquisition_value * focus_boost * (1 - scaled_dist), distance_penalty / focus_boost,
                                   node);
                } else {
                    // Penalize distant nodes
                    T penalty = std::exp(-(scaled_dist - 1));
                    new_pq.emplace(acquisition_value * penalty * penalization_factor, distance_penalty * (2 - penalty),
                                   node);
                }
            }

            pq = std::move(new_pq);
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_ENGINE_HPP
