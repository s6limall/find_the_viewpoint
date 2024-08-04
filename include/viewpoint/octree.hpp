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
#include "common/logging/logger.hpp"
#include "optimization/gpr.hpp"
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
            [[nodiscard]] bool isLeaf() const noexcept { return children[0] == nullptr; }
        };

        Octree(const Eigen::Vector3<T> &center, T size, T min_size, int max_iterations) :
            root_(std::make_unique<Node>(center, size)), min_size_(min_size), max_iterations_(max_iterations) {}

        void optimize(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                      optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
                      const ViewPoint<T> &initial_best, T target_score = 0.95) {
            best_viewpoint_ = initial_best;
            T best_score = initial_best.getScore();
            int stagnant_iterations = 0;

            for (int i = 0; i < max_iterations_; ++i) {
                if (refine(target, comparator, gpr, i)) {
                    LOG_INFO("Refinement complete at iteration {}", i);
                    break;
                }

                T current_best_score = best_viewpoint_->getScore();

                if (current_best_score >= target_score) {
                    LOG_INFO("Target score reached at iteration {}", i);
                    break;
                }

                if (current_best_score > best_score) {
                    best_score = current_best_score;
                    stagnant_iterations = 0;
                } else if (++stagnant_iterations >= patience_) {
                    LOG_INFO("Early stopping triggered after {} stagnant iterations", patience_);
                    break;
                }

                patternSearch(best_viewpoint_->getPosition(), target, comparator, gpr);

                if (i % 10 == 0) {
                    gpr.optimizeHyperparameters();
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
        static constexpr int patience_ = 20;
        static constexpr T improvement_threshold_ = 1e-8;

        bool refine(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                    optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
                    int current_iteration) {
            std::priority_queue<std::pair<T, Node *>> pq;
            pq.emplace(root_->max_ucb, root_.get());

            int nodes_explored = 0;
            const int min_nodes_to_explore = 5; // Ensure at least this many nodes are explored

            while (!pq.empty() && (nodes_explored < min_nodes_to_explore || current_iteration < max_iterations_ / 2)) {
                auto [ucb, node] = pq.top();
                pq.pop();

                if (node->size < min_size_)
                    continue;

                exploreNode(*node, target, comparator, gpr, current_iteration);
                nodes_explored++;

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
                         const std::shared_ptr<processing::image::ImageComparator> &comparator,
                         optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
                         int current_iteration) {
            if (node.points.empty()) {
                node.points = samplePoints(node);
                if (isWithinNode(node, best_viewpoint_->getPosition())) {
                    addPointsAroundBest(node);
                }
            }

            evaluatePoints(node, target, comparator, gpr, current_iteration);

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
                points.emplace_back(position);
            }

            return points;
        }

        void evaluatePoints(Node &node, const Image<> &target,
                            const std::shared_ptr<processing::image::ImageComparator> &comparator,
                            optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
                            int current_iteration) {
            node.max_ucb = std::numeric_limits<T>::lowest();

            for (auto &point: node.points) {
                if (!point.hasScore()) {
                    Image<> rendered_image = Image<>::fromViewPoint(point);
                    T score = comparator->compare(target, rendered_image);
                    point.setScore(score);

                    updateBestViewpoint(point);
                    gpr.update(point.getPosition(), score);
                }

                auto [mean, variance] = gpr.predict(point.getPosition());
                T ucb = computeUCB(mean, variance, current_iteration);
                node.max_ucb = std::max(node.max_ucb, ucb);
            }
        }

        void addPointsAroundBest(Node &node) {
            const int extra_points = 5;
            const T radius = node.size * 0.1;
            std::normal_distribution<T> dist(0, radius);
            for (int i = 0; i < extra_points; ++i) {
                Eigen::Vector3<T> offset(dist(rng_), dist(rng_), dist(rng_));
                node.points.emplace_back(best_viewpoint_->getPosition() + offset);
            }
        }

        bool isWithinNode(const Node &node, const Eigen::Vector3<T> &position) const {
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

        void patternSearch(const Eigen::Vector3<T> &center, const Image<> &target,
                           const std::shared_ptr<processing::image::ImageComparator> &comparator,
                           optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr) {
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

                        ViewPoint<T> new_viewpoint(new_point);
                        Image<> rendered_image = Image<>::fromViewPoint(new_viewpoint);
                        T score = comparator->compare(target, rendered_image);
                        new_viewpoint.setScore(score);

                        updateBestViewpoint(new_viewpoint);
                        gpr.update(new_point, score);

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
    };

} // namespace viewpoint

#endif // VIEWPOINT_OCTREE_HPP
