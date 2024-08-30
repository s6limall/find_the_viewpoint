// File: optimization/viewpoint_optimizer.hpp

#ifndef VIEWPOINT_OPTIMIZER_HPP
#define VIEWPOINT_OPTIMIZER_HPP
#include "acquisition.hpp"
#include "cache/viewpoint_cache.hpp"
#include "evaluation/viewpoint_evaluator.hpp"
#include "local_optimizer.hpp"
#include "radius_optimizer.hpp"
#include "sampling/viewpoint_sampler.hpp"
#include "spatial/octree.hpp"

namespace optimization {
    template<FloatingPoint T = double>
    class ViewpointOptimizer {
    public:
        ViewpointOptimizer(const Eigen::Vector3<T> &center, T size, T min_size, const int max_iterations,
                           optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
                           std::optional<T> radius = std::nullopt, std::optional<T> tolerance = std::nullopt) :
            octree_(center, size, min_size), sampler_(center, radius.value_or(0), tolerance.value_or(0)),
            min_size_(min_size), max_iterations_(max_iterations), gpr_(gpr),
            patience_(config::get("optimization.patience", 10)),
            improvement_threshold_(config::get("optimization.improvement_threshold", 1e-4)), radius_(radius),
            tolerance_(tolerance), recent_scores_(), target_score_(),
            cache_(typename cache::ViewpointCache<T>::CacheConfig{}),
            acquisition_(typename optimization::Acquisition<T>::Config{}),
            evaluator_(gpr, cache_, acquisition_, patience_, improvement_threshold_) {}

        void optimize(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                      const ViewPoint<T> &initial_best, T target_score = T(0.95)) {
            LOG_INFO("Starting optimization with target score {}", target_score);
            target_score_ = target_score;
            best_viewpoint_ = initial_best;
            T best_score = initial_best.getScore();
            stagnant_iterations_ = 0;
            recent_scores_.clear();

            local_optimizer_ = std::make_unique<LocalOptimizer<T>>(comparator);

            typename optimization::Acquisition<T>::Config acquisition_config(
                    optimization::Acquisition<T>::Strategy::ADAPTIVE,
                    config::get("optimization.gp.acquisition.beta", 2.0),
                    config::get("optimization.gp.acquisition.exploration_weight", 1.0),
                    config::get("optimization.gp.acquisition.exploitation_weight", 1.0),
                    config::get("optimization.gp.acquisition.momentum", 0.1));

            const auto local_search_frequency = config::get("optimization.local_search_frequency", 10);
            const auto hyperparameter_optimization_frequency =
                    config::get("optimization.gp.kernel.hyperparameters.optimization_frequency", 10);

            acquisition_.updateConfig(acquisition_config);
            const int max_stagnant_iterations = patience_;
            for (int i = 0; i < max_iterations_; ++i) {
                if (refine(target, comparator, i)) {
                    LOG_INFO("Refinement complete at iteration {}", i);
                    break;
                }

                if (!best_viewpoint_) {
                    LOG_ERROR("Lost best viewpoint during optimization");
                    return;
                }

                T current_best_score = best_viewpoint_->getScore();

                if (current_best_score > best_score) {
                    best_score = current_best_score;
                    stagnant_iterations_ = 0;
                } else {
                    ++stagnant_iterations_;
                }

                if (i % local_search_frequency == 0) {
                    localRefinement(target, comparator);
                    /*LOG_INFO("Initiating local search at iteration [{} / {}]", i, max_iterations_);
                    best_viewpoint_ = local_optimizer_->optimize(best_viewpoint_.value(), target);
                    current_best_score = best_viewpoint_->getScore();
                    if (current_best_score > best_score) {
                        best_score = current_best_score;
                        stagnant_iterations_ = 0;
                        LOG_INFO("Local optimization improved score to {}", best_score);
                    }*/
                }

                if (evaluator_.hasConverged(current_best_score, best_score, target_score, i, recent_scores_,
                                            stagnant_iterations_, *best_viewpoint_) ||
                    stagnant_iterations_ >= max_stagnant_iterations) {
                    LOG_INFO("Optimization converged or stagnated at iteration {}", i);
                    break;
                }

                if (i % hyperparameter_optimization_frequency == 0) {
                    gpr_.optimizeHyperparameters();
                }
            }

            if (!best_viewpoint_) {
                LOG_ERROR("No best viewpoint found after main optimization");
                return;
            }

            LOG_INFO("Main optimization complete. Best viewpoint before final refinement: {}",
                     best_viewpoint_->toString());

            // Perform final local optimization
            /*best_viewpoint_ = local_optimizer_->optimize(*best_viewpoint_, target);*/

            LOG_INFO("Best viewpoint after final local optimization: {}", best_viewpoint_->toString());

            auto refined_result = optimizeRadius(target);

            if (!refined_result) {
                LOG_ERROR("Radius refinement failed");
                return;
            }

            Image<T> refined_image = Image<>::fromViewPoint(refined_result->best_viewpoint);

            T refined_score = comparator->compare(target, refined_image);
            refined_result->best_viewpoint.setScore(refined_score);
            refined_image.setScore(refined_score);
            LOG_INFO("Refined viewpoint: {}, Score: {}", refined_image.getViewPoint()->toString(), refined_score);
            LOG_INFO("Optimization complete.");
            LOG_INFO("Initial best viewpoint: {}", initial_best.toString());
            LOG_INFO("Best viewpoint after main optimization: {}", best_viewpoint_->toString());
            LOG_INFO("Final best viewpoint after radius refinement: {}", refined_result->best_viewpoint.toString());
            LOG_INFO("Initial score: {:.6f}", initial_best.getScore());
            LOG_INFO("Total score improvement: {:.6f}", refined_score - initial_best.getScore());
            LOG_INFO("Radius refinement iterations: {}", refined_result->iterations);

            if (refined_score > best_viewpoint_->getScore()) {
                best_viewpoint_ = refined_result->best_viewpoint;
                LOG_INFO("Radius refinement improved the viewpoint");
            } else {
                LOG_INFO("Radius refinement did not improve the viewpoint. Keeping the original.");
            }
        }

        [[nodiscard]] std::optional<ViewPoint<T>> getBestViewpoint() const noexcept { return best_viewpoint_; }

    private:
        spatial::Octree<T> octree_;
        ViewpointSampler<T> sampler_;
        T min_size_;
        int max_iterations_;
        optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr_;
        std::optional<ViewPoint<T>> best_viewpoint_;
        int patience_;
        const T improvement_threshold_;
        std::optional<T> radius_, tolerance_;
        std::deque<T> recent_scores_;
        int stagnant_iterations_ = 0;
        T target_score_;
        cache::ViewpointCache<T> cache_;
        mutable optimization::Acquisition<T> acquisition_;
        ViewpointEvaluator<T> evaluator_;
        std::unique_ptr<LocalOptimizer<T>> local_optimizer_;

        void localRefinement(const Image<> &target,
                             const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            if (!best_viewpoint_)
                return;

            const int max_iterations = config::get("optimization.local_search.max_iterations", 10);
            const T learning_rate = config::get("optimization.local_search.learning_rate", 0.01);
            const T epsilon = config::get("optimization.local_search.epsilon", 1e-5);

            Eigen::Vector3<T> current_position = best_viewpoint_->getPosition();
            T current_score = best_viewpoint_->getScore();

            for (int i = 0; i < max_iterations; ++i) {
                Eigen::Vector3<T> gradient;
                for (int j = 0; j < 3; ++j) {
                    Eigen::Vector3<T> perturbed_position = current_position;
                    perturbed_position[j] += epsilon;

                    ViewPoint<T> perturbed_viewpoint(perturbed_position);
                    Image<> perturbed_image = Image<>::fromViewPoint(perturbed_viewpoint);
                    T perturbed_score = comparator->compare(target, perturbed_image);

                    gradient[j] = (perturbed_score - current_score) / epsilon;

                    // Store metrics for perturbed points during local refinement
                    metrics::recordMetrics(perturbed_viewpoint, {{"position_x", perturbed_position.x()},
                                                                 {"position_y", perturbed_position.y()},
                                                                 {"position_z", perturbed_position.z()},
                                                                 {"score", perturbed_score},
                                                                 {"refinement_iteration", i}});
                }

                Eigen::Vector3<T> new_position = current_position + learning_rate * gradient;
                ViewPoint<T> new_viewpoint(new_position);
                Image<> new_image = Image<>::fromViewPoint(new_viewpoint);
                T new_score = comparator->compare(target, new_image);

                if (new_score > current_score) {
                    current_position = new_position;
                    current_score = new_score;
                    updateBestViewpoint(ViewPoint<T>(new_position, new_score));
                } else {
                    break;
                }
            }
        }

        bool refine(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                    int current_iteration) {
            updateAcquisitionFunction(current_iteration);

            std::priority_queue<NodeScore> pq;
            T distance_scale = octree_.getRoot()->size / 10.0;
            pq.emplace(octree_.getRoot()->max_acquisition, 0.0, octree_.getRoot());

            int nodes_explored = 0;
            const int min_nodes_to_explore = config::get("octree.min_nodes_to_explore", 5);
            T best_score_this_refinement = best_viewpoint_->getScore();

            while (!pq.empty() && (nodes_explored < min_nodes_to_explore || current_iteration < max_iterations_ / 2)) {
                auto [acquisition_value, distance_penalty, node] = pq.top();
                pq.pop();

                if (node->size < min_size_)
                    continue;

                exploreNode(*node, target, comparator);
                nodes_explored++;

                T current_best_score = best_viewpoint_->getScore();
                if (evaluator_.hasConverged(current_best_score, best_score_this_refinement, target_score_,
                                            current_iteration, recent_scores_, stagnant_iterations_,
                                            *best_viewpoint_)) {
                    LOG_INFO("Convergence detected during refinement at iteration {}", current_iteration);
                    return true;
                }


                if (current_best_score > best_score_this_refinement) {
                    best_score_this_refinement = current_best_score;
                    LOG_INFO("Best score updated to {} at iteration {} - {} nodes explored.",
                             best_score_this_refinement, current_iteration, nodes_explored);

                    localRefinement(target, comparator);
                }

                if (!node->isLeaf()) {
                    for (auto &child: node->children) {
                        if (child) {
                            T dist_to_best = (child->center - best_viewpoint_->getPosition()).norm();
                            T child_distance_penalty = std::exp(-dist_to_best / distance_scale);
                            pq.emplace(child->max_acquisition, child_distance_penalty, child.get());
                        }
                    }
                }
            }

            return nodes_explored == 0;
        }

        void exploreNode(typename spatial::Octree<T>::Node &node, const Image<> &target,
                         const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            if (node.points.empty()) {
                node.points = sampler_.samplePoints(node);
                if (octree_.isWithinNode(node, best_viewpoint_->getPosition())) {
                    sampler_.addPointsAroundBest(node, *best_viewpoint_);
                }
            }

            node.max_acquisition = std::numeric_limits<T>::lowest();
            for (auto &point: node.points) {
                evaluator_.evaluatePoint(point, target, comparator);
                auto [mean, std_dev] = gpr_.predict(point.getPosition());
                T acquisition_value = evaluator_.computeAcquisition(point.getPosition(), mean, std_dev);
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
            }

            node.explored = true;
        }

        void updateBestViewpoint(const ViewPoint<T> &point) {
            if (!best_viewpoint_ || point.getScore() > best_viewpoint_->getScore()) {
                best_viewpoint_ = point;
                LOG_INFO("New best viewpoint: {} with score {}", point.toString(), point.getScore());
            }
        }

        void updateAcquisitionFunction(int current_iteration) {
            T recent_improvement_rate = calculateRecentImprovementRate();
            T current_best_score = best_viewpoint_->getScore();

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

        std::optional<typename RadiusOptimizer<T>::RadiusOptimizerResult> optimizeRadius(const Image<> &target) const {
            if (!best_viewpoint_) {
                LOG_WARN("No best viewpoint found for final radius refinement");
                return std::nullopt;
            }

            auto renderFunction = [](const ViewPoint<T> &vp) { return Image<T>::fromViewPoint(vp); };

            auto radius_optimizer = RadiusOptimizer<T>();
            auto result = radius_optimizer.optimize(*best_viewpoint_, target, renderFunction);

            LOG_INFO("Final radius refinement complete.");

            return result;
        }

        struct NodeScore {
            T acquisition_value;
            T distance_penalty;
            typename spatial::Octree<T>::Node *node;

            NodeScore(T acquisition, T dist_penalty, typename spatial::Octree<T>::Node *n) :
                acquisition_value(acquisition), distance_penalty(dist_penalty), node(n) {}

            bool operator<(const NodeScore &other) const {
                // Higher values have higher priority
                return (acquisition_value - distance_penalty) < (other.acquisition_value - other.distance_penalty);
            }
        };
    };
} // namespace optimization

#endif // VIEWPOINT_OPTIMIZER_HPP
