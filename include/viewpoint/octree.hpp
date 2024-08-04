// File: viewpoint/octree.hpp

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

        Octree(const Eigen::Vector3<T> &center, T size, T min_size) :
            root_(std::make_unique<Node>(center, size)), min_size_(min_size) {}

        void optimize(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                      optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
                      const ViewPoint<T> &initial_best, int max_iterations = 100, T target_score = 0.95) {
            best_viewpoint_ = initial_best;
            T previous_best_score = best_viewpoint_->getScore();

            for (int i = 0; i < max_iterations; ++i) {
                if (refine(target, comparator, gpr)) {
                    LOG_INFO("Refinement complete at iteration {}", i);
                    break;
                }

                T current_best_score = best_viewpoint_->getScore();

                if (current_best_score >= target_score) {
                    LOG_INFO("Target score reached at iteration {}", i);
                    break;
                }

                if (current_best_score - previous_best_score < improvement_threshold_) {
                    stagnant_count_++;
                    if (stagnant_count_ >= max_stagnant_iterations_) {
                        LOG_INFO("Optimization stagnated for {} iterations. Stopping early.", max_stagnant_iterations_);
                        break;
                    }
                } else {
                    stagnant_count_ = 0;
                    previous_best_score = current_best_score;
                }

                // Global search
                auto candidates = getBestCandidates(10);
                for (auto &candidate: candidates) {
                    T ei = gpr.expectedImprovement(candidate.getPosition(), best_viewpoint_->getScore());
                    if (ei > 0) {
                        evaluateAndUpdatePoint(candidate, target, comparator, gpr);
                    }
                }

                // Local search around best viewpoint
                localSearch(best_viewpoint_->getPosition(), target, comparator, gpr);

                // Periodically optimize GPR hyperparameters
                if (i % 10 == 0) {
                    gpr.optimizeHyperparameters();
                }
            }
        }

        std::vector<ViewPoint<T>> getBestCandidates(size_t n) const {
            std::vector<ViewPoint<T>> candidates;
            std::function<void(const Node *)> traverse = [&](const Node *node) {
                if (node->isLeaf()) {
                    candidates.insert(candidates.end(), node->points.begin(), node->points.end());
                } else {
                    for (const auto &child: node->children) {
                        if (child)
                            traverse(child.get());
                    }
                }
            };
            traverse(root_.get());

            std::partial_sort(candidates.begin(), candidates.begin() + std::min(n, candidates.size()), candidates.end(),
                              [](const auto &a, const auto &b) { return a.getScore() > b.getScore(); });
            candidates.resize(std::min(n, candidates.size()));
            return candidates;
        }

        [[nodiscard]] std::optional<ViewPoint<T>> getBestViewpoint() const noexcept { return best_viewpoint_; }

        void evaluateAndUpdatePoint(ViewPoint<T> &point, const Image<> &target,
                                    const std::shared_ptr<processing::image::ImageComparator> &comparator,
                                    optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr) {
            if (!point.hasScore()) {
                Image<> rendered_image = Image<>::fromViewPoint(point);
                T score = comparator->compare(target, rendered_image);
                point.setScore(score);

                updateBestViewpoint(point);
                gpr.update(point.getPosition(), score);
            }
        }

        void localSearch(const Eigen::Vector3<T> &center, const Image<> &target,
                         const std::shared_ptr<processing::image::ImageComparator> &comparator,
                         optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr) {
            const T search_radius = min_size_ * 2;
            std::uniform_real_distribution<T> dist(-search_radius, search_radius);

            for (int i = 0; i < 10; ++i) {
                Eigen::Vector3<T> offset(dist(rng_), dist(rng_), dist(rng_));
                ViewPoint<T> new_point(center + offset);
                evaluateAndUpdatePoint(new_point, target, comparator, gpr);
            }
        }

    private:
        std::unique_ptr<Node> root_;
        T min_size_;
        std::optional<ViewPoint<T>> best_viewpoint_;
        std::mt19937 rng_{std::random_device{}()};
        // early stopping:
        const int max_stagnant_iterations_ = 50;
        const T improvement_threshold_ = 1e-8;
        int stagnant_count_ = 0;


        bool refine(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                    optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr) {
            std::priority_queue<std::pair<T, Node *>> pq;
            pq.emplace(root_->max_ucb, root_.get());

            bool refined = false;
            while (!pq.empty()) {
                auto [_, node] = pq.top();
                pq.pop();

                if (node->size < min_size_ || node->explored)
                    continue;

                exploreNode(*node, target, comparator, gpr);
                refined = true;

                if (!node->isLeaf()) {
                    for (auto &child: node->children) {
                        if (child)
                            pq.emplace(child->max_ucb, child.get());
                    }
                }
            }

            return !refined; // Return true if no refinement occurred (convergence)
        }

        void exploreNode(Node &node, const Image<> &target,
                         const std::shared_ptr<processing::image::ImageComparator> &comparator,
                         optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr) {
            if (node.points.empty()) {
                node.points = samplePoints(node);
                // Add extra points around the best viewpoint if it's within this node
                if (isWithinNode(node, best_viewpoint_->getPosition())) {
                    addPointsAroundBest(node);
                }
            }

            evaluatePoints(node, target, comparator, gpr);

            if (shouldSplit(node)) {
                splitNode(node);
            }

            node.explored = true;
        }

        std::vector<ViewPoint<T>> samplePoints(const Node &node) {
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
                            optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr) {
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
                T ucb = computeUCB(mean, variance);
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
                    (best_viewpoint_ && node.center.isApprox(best_viewpoint_->getPosition(), node.size)));
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

        T computeUCB(T mean, T variance) const {
            static const T exploration_factor = 2.0;
            return mean + exploration_factor * std::sqrt(variance);
        }
    };

} // namespace viewpoint

#endif // VIEWPOINT_OCTREE_HPP
