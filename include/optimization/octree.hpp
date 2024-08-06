#ifndef VIEWPOINT_OCTREE_HPP
#define VIEWPOINT_OCTREE_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <concepts>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <vector>
#include "cache/viewpoint_cache.hpp"
#include "common/logging/logger.hpp"
#include "optimization/gpr.hpp"
#include "optimization/levenberg_marquardt.hpp"
#include "processing/image/comparator.hpp"
#include "types/viewpoint.hpp"

namespace viewpoint {

    template<typename T>
    concept FloatingPoint = std::is_floating_point_v<T>;

    template<FloatingPoint T = double>
    class Octree {
    public:
        struct Node {
            Eigen::Vector3<T> center;
            T size;
            std::vector<ViewPoint<T>> points;
            std::array<std::unique_ptr<Node>, 8> children;
            bool explored = false;
            T max_ucb = std::numeric_limits<T>::lowest();

            Node(const Eigen::Vector3<T> &center, T size) : center(center), size(size) {}
            [[nodiscard]] bool isLeaf() const noexcept { return children[0] == nullptr; } // node without children
        };

        Octree(const Eigen::Vector3<T> &center, T size, T min_size, int max_iterations,
               optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
               std::optional<T> radius = std::nullopt, std::optional<T> tolerance = std::nullopt) :
            root_(std::make_unique<Node>(center, size)), min_size_(min_size), max_iterations_(max_iterations),
            gpr_(gpr), radius_(radius), tolerance_(tolerance), recent_scores_(), target_score_(),
            cache_(typename cache::ViewpointCache<T>::CacheConfig{}) {}


        void optimize(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                      const ViewPoint<T> &initial_best, T target_score = T(0.95)) {
            LOG_INFO("Starting optimization with target score {}", target_score);
            target_score_ = target_score;
            best_viewpoint_ = initial_best;
            T best_score = initial_best.getScore();
            stagnant_iterations_ = 0;
            recent_scores_.clear();

            for (int i = 0; i < max_iterations_; ++i) {
                if (refine(target, comparator, i)) {
                    LOG_INFO("Refinement complete at iteration {}", i);
                    break;
                }

                T current_best_score = best_viewpoint_->getScore();

                if (hasConverged(current_best_score, best_score, target_score, i)) {
                    break;
                }

                if (current_best_score > best_score) {
                    best_score = current_best_score;
                }

                patternSearch(best_viewpoint_->getPosition(), target, comparator);

                if (i % 10 == 0) {
                    gpr_.optimizeHyperparameters();
                }
            }
        }

        [[nodiscard]] std::optional<ViewPoint<T>> getBestViewpoint() const noexcept { return best_viewpoint_; }

    private:
        std::unique_ptr<Node> root_;
        T min_size_;
        int max_iterations_;
        std::optional<ViewPoint<T>> best_viewpoint_;
        mutable std::mt19937 rng_{std::random_device{}()};
        static constexpr int patience_ = 10;
        static constexpr T improvement_threshold_ = 1e-4;
        std::optional<T> radius_, tolerance_;
        optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr_;
        std::deque<T> recent_scores_;
        int stagnant_iterations_ = 0;
        static constexpr int window_size_ = 5;
        double target_score_;
        cache::ViewpointCache<T> cache_;


        bool refine(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                    int current_iteration) {
            std::priority_queue<std::pair<T, Node *>> pq;
            pq.emplace(root_->max_ucb, root_.get());

            int nodes_explored = 0;
            const int min_nodes_to_explore = 5; // Ensure at least this many nodes are explored
            T best_score_this_refinement = best_viewpoint_->getScore();

            while (!pq.empty() && (nodes_explored < min_nodes_to_explore || current_iteration < max_iterations_ / 2)) {
                auto [ucb, node] = pq.top();
                pq.pop();

                if (node->size < min_size_)
                    continue;

                exploreNode(*node, target, comparator, current_iteration);
                nodes_explored++;

                // Check for convergence after exploring each node
                T current_best_score = best_viewpoint_->getScore();
                if (hasConverged(current_best_score, best_score_this_refinement, target_score_, current_iteration)) {
                    LOG_INFO("Convergence detected during refinement at iteration {}", current_iteration);
                    return true;
                }

                // Update best score for this refinement step
                if (current_best_score > best_score_this_refinement) {
                    best_score_this_refinement = current_best_score;
                    LOG_INFO("Best score updated to {} at iteration {} - {} nodes explored.",
                             best_score_this_refinement, current_iteration, nodes_explored);
                }

                if (!node->isLeaf()) {
                    for (auto &child: node->children) {
                        if (child)
                            pq.emplace(child->max_ucb, child.get());
                    }
                }
            }

            return nodes_explored == 0; // Return true if no nodes were explored
        }

        void exploreNode(Node &node, const Image<> &target,
                         const std::shared_ptr<processing::image::ImageComparator> &comparator, int current_iteration) {
            if (node.points.empty()) {
                node.points = samplePoints(node);
                if (isWithinNode(node, best_viewpoint_->getPosition())) {
                    addPointsAroundBest(node);
                }
            }

            evaluatePoints(node, target, comparator, current_iteration);

            if (shouldSplit(node)) {
                splitNode(node);
            }

            node.explored = true;
        }

        std::vector<ViewPoint<T>> samplePoints(const Node &node) const {
            std::vector<ViewPoint<T>> points;
            std::uniform_real_distribution<T> dist(-0.5, 0.5);

            points.reserve(10);
            for (int i = 0; i < 10; ++i) {
                Eigen::Vector3<T> position =
                        node.center + node.size * Eigen::Vector3<T>(dist(rng_), dist(rng_), dist(rng_));
                position = projectToShell(position);
                points.emplace_back(position);
            }

            return points;
        }

        void evaluatePoints(Node &node, const Image<> &target,
                            const std::shared_ptr<processing::image::ImageComparator> &comparator,
                            int current_iteration) {
            node.max_ucb = std::numeric_limits<T>::lowest();

            for (auto &point: node.points) {
                if (!point.hasScore()) {
                    auto cached_score = cache_.query(point.getPosition());
                    if (cached_score) {
                        point.setScore(*cached_score);
                        LOG_DEBUG("Using cached score {} for position {}", *cached_score, point.getPosition());
                    } else {
                        Image<> rendered_image = Image<>::fromViewPoint(point);
                        T score = comparator->compare(target, rendered_image);
                        point.setScore(score);
                        cache_.insert(point);
                        LOG_DEBUG("Computed new score {} for position {}", score, point.getPosition());
                    }

                    updateBestViewpoint(point);
                    gpr_.update(point.getPosition(), point.getScore());
                } else {
                    // Update the cache with the existing point
                    cache_.update(point);
                }

                auto [mean, variance] = gpr_.predict(point.getPosition());
                T ucb = computeUCB(mean, variance, current_iteration);
                node.max_ucb = std::max(node.max_ucb, ucb);
            }
        }

        void addPointsAroundBest(Node &node) {
            constexpr int extra_points = 6;
            const T radius = node.size * 0.1;
            std::normal_distribution<T> dist(0, radius);
            for (int i = 0; i < extra_points; ++i) {
                Eigen::Vector3<T> offset(dist(rng_), dist(rng_), dist(rng_));
                node.points.emplace_back(best_viewpoint_->getPosition() + offset);
            }
        }

        static bool isWithinNode(const Node &node, const Eigen::Vector3<T> &position) {
            return (position - node.center).cwiseAbs().maxCoeff() <= node.size / 2;
        }

        void updateBestViewpoint(const ViewPoint<T> &point) {
            if (!best_viewpoint_ || point.getScore() > best_viewpoint_->getScore()) {
                best_viewpoint_ = point;
                LOG_INFO("New best viewpoint: {} with score {}", point.toString(), point.getScore());
            }
        }

        [[nodiscard]] bool shouldSplit(const Node &node) const noexcept {
            return node.size > min_size_ * T(2) &&
                   (node.points.size() > 10 ||
                    (best_viewpoint_ && (node.center - best_viewpoint_->getPosition()).norm() < node.size));
        }

        void splitNode(Node &node) {
            T child_size = node.size / T(2);

            LOG_INFO("Splitting node at {} with size {}", node.center, node.size);
            for (int i = 0; i < 8; ++i) {
                Eigen::Vector3<T> offset((i & 1) ? child_size : -child_size, (i & 2) ? child_size : -child_size,
                                         (i & 4) ? child_size : -child_size);
                node.children[i] = std::make_unique<Node>(node.center + offset * T(0.5), child_size);
            }

            for (const auto &point: node.points) {
                int index = 0;
                for (int d = 0; d < 3; ++d) {
                    if (point.getPosition()[d] > node.center[d]) {
                        index |= (1 << d);
                    }
                }
                node.children[index]->points.push_back(point);
            }

            node.points.clear();
            node.points.shrink_to_fit();
        }

        T computeUCB(T mean, T variance, int current_iteration) const {
            T exploration_factor = 2.0 * std::exp(-current_iteration / static_cast<T>(max_iterations_));
            return mean + exploration_factor * std::sqrt(variance);
        }

        // Local search - precise refinement
        void patternSearch(const Eigen::Vector3<T> &center, const Image<> &target,
                           const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            const T initial_step = min_size_ * 0.1;
            T step = initial_step;
            const int max_pattern_iterations = 20;
            const T tolerance = 1e-6;

            Eigen::Vector3<T> best_point = center;
            T best_score = best_viewpoint_->getScore();

            for (int iter = 0; iter < max_pattern_iterations; ++iter) {
                bool improved = false;
                for (int dim = 0; dim < 3; ++dim) {
                    for (int direction: {-1, 1}) {
                        Eigen::Vector3<T> new_point = best_point;
                        new_point[dim] += direction * step;

                        auto cached_score = cache_.query(new_point);
                        T score;
                        if (cached_score) {
                            score = *cached_score;
                            LOG_DEBUG("Using cached score {} for position {}", score, new_point);
                        } else {
                            ViewPoint<T> new_viewpoint(new_point);
                            Image<> rendered_image = Image<>::fromViewPoint(new_viewpoint);
                            score = comparator->compare(target, rendered_image);
                            new_viewpoint.setScore(score);
                            cache_.update(new_viewpoint);
                            LOG_DEBUG("Computed new score {} for position {}", score, new_point);
                        }

                        updateBestViewpoint(ViewPoint<T>(new_point, score));
                        gpr_.update(new_point, score);

                        if (score > best_score) {
                            best_point = new_point;
                            best_score = score;
                            improved = true;
                        }
                    }
                }

                if (!improved) {
                    step *= 0.5;
                }

                if (step < tolerance) {
                    break;
                }
            }
        }

        Eigen::Vector3<T> projectToShell(const Eigen::Vector3<T> &point) const {
            if (!radius_ || !tolerance_)
                return point;

            Eigen::Vector3<T> direction = point - root_->center;
            T distance = direction.norm();
            T min_radius = *radius_ * (1 - *tolerance_);
            T max_radius = *radius_ * (1 + *tolerance_);

            if (distance < min_radius) {
                return root_->center + direction.normalized() * min_radius;
            } else if (distance > max_radius) {
                return root_->center + direction.normalized() * max_radius;
            }
            return point;
        }

        bool hasConverged(T current_score, T best_score, T target_score, int current_iteration) {
            // Check if we've reached or exceeded the target score
            if (current_score >= target_score) {
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

            // Moving average convergence check
            recent_scores_.push_back(current_score);
            if (recent_scores_.size() > window_size_) {
                recent_scores_.pop_front();
            }

            if (recent_scores_.size() == window_size_) {
                T avg_score = std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0)) / window_size_;
                T score_variance =
                        std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0),
                                        [avg_score](T acc, T score) { return acc + std::pow(score - avg_score, 2); }) /
                        window_size_;

                if (score_variance < T(1e-6) && avg_score > target_score * T(0.95)) {
                    LOG_INFO("Convergence detected based on moving average at iteration {}", current_iteration);
                    return true;
                }
            }

            // Check confidence interval using GPR
            auto [mean, variance] = gpr_.predict(best_viewpoint_->getPosition());
            T confidence_interval = T(1.96) * std::sqrt(variance); // 95% confidence interval

            if (mean - confidence_interval > target_score) {
                LOG_INFO("High confidence in solution at iteration {}", current_iteration);
                return true;
            }

            return false;
        }

        void levenbergMarquardtRefinement(const Image<> &target,
                                          const std::shared_ptr<processing::image::ImageComparator> &comparator,
                                          optimization::LevenbergMarquardt<T, 3> &lm_optimizer) {
            auto error_func = [&](const Eigen::Vector3<T> &position) {
                auto cached_score = cache_.query(position);
                if (cached_score) {
                    LOG_DEBUG("Using cached score {} for position {}", *cached_score, position);
                    return static_cast<T>(1.0) - *cached_score;
                }
                ViewPoint<T> viewpoint(position);
                Image<> rendered_image = Image<>::fromViewPoint(viewpoint);
                T score = comparator->compare(target, rendered_image);
                cache_.update(ViewPoint<T>(position, score));
                LOG_DEBUG("Computed new score {} for position {}", score, position);
                return static_cast<T>(1.0) - score;
            };

            auto jacobian_func = [&](const Eigen::Vector3<T> &position) {
                const T h = static_cast<T>(1e-5);
                Eigen::Matrix<T, 1, 3> J;

                for (int i = 0; i < 3; ++i) {
                    Eigen::Vector3<T> perturbed_position = position;
                    perturbed_position[i] += h;
                    J(0, i) = (error_func(perturbed_position) - error_func(position)) / h;
                }

                return J;
            };

            auto result = lm_optimizer.optimize(best_viewpoint_->getPosition(), error_func, jacobian_func);

            if (result) {
                ViewPoint<T> new_viewpoint(result->position, static_cast<T>(1.0) - result->final_error);
                updateBestViewpoint(new_viewpoint);
                gpr_.update(result->position, new_viewpoint.getScore());
            }
        }
    };

} // namespace viewpoint

#endif // VIEWPOINT_OCTREE_HPP
