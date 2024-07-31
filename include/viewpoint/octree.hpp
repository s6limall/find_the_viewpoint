// File: viewpoint/octree.hpp

#ifndef VIEWPOINT_OCTREE_HPP
#define VIEWPOINT_OCTREE_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <vector>
#include "common/logging/logger.hpp"
#include "optimization/gpr.hpp"
#include "processing/image/comparator.hpp"
#include "types/viewpoint.hpp"

namespace viewpoint {

    struct EigenMatrixComparator {
        bool operator()(const Eigen::Matrix<double, 3, 1> &a, const Eigen::Matrix<double, 3, 1> &b) const {
            if (a(0, 0) != b(0, 0))
                return a(0, 0) < b(0, 0);
            if (a(1, 0) != b(1, 0))
                return a(1, 0) < b(1, 0);
            return a(2, 0) < b(2, 0);
        }
    };

    template<typename T = double>
    class Octree {
    public:
        struct Node {
            Eigen::Matrix<T, 3, 1> center;
            T size;
            std::vector<ViewPoint<T>> points;
            std::array<std::unique_ptr<Node>, 8> children;
            bool explored;
            T similarity_gradient;

            Node(const Eigen::Matrix<T, 3, 1> &center, T size) noexcept :
                center(center), size(size), points(), children{}, explored(false), similarity_gradient(0) {}
            bool isLeaf() const noexcept { return children[0] == nullptr; }
        };

        Octree(const Eigen::Matrix<T, 3, 1> &origin, T size, T resolution, size_t max_points_per_node = 10) noexcept :
            resolution_(resolution), max_points_per_node_(max_points_per_node),
            root_(std::make_unique<Node>(origin, size)) {}

        void insert(const ViewPoint<T> &point);
        void refine(int max_depth, const Image<> &target,
                    const std::shared_ptr<processing::image::ImageComparator> &comparator,
                    const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr);
        std::vector<ViewPoint<T>> sampleNewViewpoints(
                size_t n, const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) const;
        bool checkConvergence() const;
        void failureRecovery();

    private:
        std::unique_ptr<Node> root_;
        T resolution_;
        size_t max_points_per_node_;
        std::set<Eigen::Matrix<T, 3, 1>, EigenMatrixComparator> explored_points_;
        int stuck_count_ = 0;
        int max_stuck_iterations_ = 10;
        T best_score_ = std::numeric_limits<T>::lowest();

        void insert(Node *node, const ViewPoint<T> &point);
        void refineNode(Node &node, int depth, int max_depth, const Image<> &target,
                        const std::shared_ptr<processing::image::ImageComparator> &comparator,
                        const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr);
        bool isWithinBounds(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept;
        size_t getOctant(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept;
        void subdivide(Node &node);
        void traverseTree(const std::function<void(const Node &)> &visit) const;
        bool isExplored(const Eigen::Matrix<T, 3, 1> &point) const noexcept;
        T computeSimilarityGradient(const Node &node, const Image<> &target,
                                    const std::shared_ptr<processing::image::ImageComparator> &comparator) const;
        T getBestScore() const;
        void exploreDiscardedRegions();
    };

    template<typename T>
    void Octree<T>::insert(const ViewPoint<T> &point) {
        if (isExplored(point.getPosition())) {
            LOG_WARN("Point ({}, {}, {}) has already been explored.", point.getPosition().x(), point.getPosition().y(),
                     point.getPosition().z());
            return;
        }
        insert(root_.get(), point);
        explored_points_.insert(point.getPosition());
    }

    template<typename T>
    void Octree<T>::insert(Node *node, const ViewPoint<T> &point) {
        if (!isWithinBounds(*node, point.getPosition())) {
            LOG_WARN("Point ({}, {}, {}) is out of bounds for this node.", point.getPosition().x(),
                     point.getPosition().y(), point.getPosition().z());
            return;
        }

        while (true) {
            if (node->points.size() < max_points_per_node_ || node->size <= resolution_) {
                node->points.push_back(point);
                LOG_DEBUG("Inserted point ({}, {}, {}) into the node.", point.getPosition().x(),
                          point.getPosition().y(), point.getPosition().z());
                return;
            }

            if (node->children[0] == nullptr) {
                subdivide(*node);
            }

            size_t octant = getOctant(*node, point.getPosition());
            node = node->children[octant].get();
        }
    }

    template<typename T>
    void
    Octree<T>::refine(int max_depth, const Image<> &target,
                      const std::shared_ptr<processing::image::ImageComparator> &comparator,
                      const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) {
        LOG_INFO("Starting Octree refinement.");
        refineNode(*root_, 0, max_depth, target, comparator, gpr);
    }

    template<typename T>
    void
    Octree<T>::refineNode(Node &node, int depth, int max_depth, const Image<> &target,
                          const std::shared_ptr<processing::image::ImageComparator> &comparator,
                          const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) {
        if (depth >= max_depth || node.explored)
            return;

        for (auto &point: node.points) {
            if (!point.hasScore()) {
                auto similarity_score =
                        comparator->compare(target.getImage(), Image<T>::fromViewPoint(point).getImage());
                point.setScore(similarity_score);
            }

            if (!point.hasUncertainty()) {
                auto [mean, variance] = gpr.predict(point.getPosition());
                point.setUncertainty(variance);
            }
        }

        node.similarity_gradient = computeSimilarityGradient(node, target, comparator);

        if (node.isLeaf() && (node.points.size() > max_points_per_node_ || node.similarity_gradient > 0.1) &&
            node.size > resolution_) {
            subdivide(node);
            LOG_DEBUG("Subdivided node at depth {} with center ({}, {}, {})", depth, node.center.x(), node.center.y(),
                      node.center.z());
        }

        for (auto &child: node.children) {
            if (child) {
                refineNode(*child, depth + 1, max_depth, target, comparator, gpr);
            }
        }
        node.explored = true;
    }

    template<typename T>
    std::vector<ViewPoint<T>> Octree<T>::sampleNewViewpoints(
            size_t n, const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
            const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) const {
        std::vector<ViewPoint<T>> new_viewpoints;
        std::vector<std::pair<ViewPoint<T>, T>> weighted_points;

        traverseTree([&](const Node &node) {
            for (const auto &point: node.points) {
                T weight = point.getUncertainty() * point.getScore() * node.similarity_gradient;
                weighted_points.emplace_back(point, weight);
                LOG_DEBUG("Weight for point ({}, {}, {}): {}", point.getPosition().x(), point.getPosition().y(),
                          point.getPosition().z(), weight);
            }
        });

        if (weighted_points.empty()) {
            LOG_WARN("No weighted points available for sampling.");
            return new_viewpoints;
        }

        std::vector<double> weights;
        for (const auto &wp: weighted_points) {
            weights.push_back(wp.second);
        }

        std::discrete_distribution<> dist(weights.begin(), weights.end());
        std::default_random_engine gen;

        for (size_t i = 0; i < n; ++i) {
            new_viewpoints.push_back(weighted_points[dist(gen)].first);
        }

        LOG_INFO("Sampled {} new viewpoints.", n);
        return new_viewpoints;
    }

    template<typename T>
    bool Octree<T>::isWithinBounds(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept {
        T half_size = node.size / 2;
        return (point.array() >= (node.center.array() - half_size)).all() &&
               (point.array() <= (node.center.array() + half_size)).all();
    }

    template<typename T>
    size_t Octree<T>::getOctant(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept {
        size_t octant = 0;
        if (point.x() >= node.center.x())
            octant |= 1;
        if (point.y() >= node.center.y())
            octant |= 2;
        if (point.z() >= node.center.z())
            octant |= 4;
        return octant;
    }

    template<typename T>
    void Octree<T>::subdivide(Node &node) {
        for (size_t i = 0; i < 8; ++i) {
            Eigen::Matrix<T, 3, 1> new_center = node.center;
            T offset = node.size / 4;
            if (i & 1)
                new_center.x() += offset;
            else
                new_center.x() -= offset;
            if (i & 2)
                new_center.y() += offset;
            else
                new_center.y() -= offset;
            if (i & 4)
                new_center.z() += offset;
            else
                new_center.z() -= offset;
            node.children[i] = std::make_unique<Node>(new_center, node.size / 2);
        }

        for (const auto &point: node.points) {
            size_t octant = getOctant(node, point.getPosition());
            node.children[octant]->points.push_back(point);
        }
        node.points.clear();
    }

    template<typename T>
    void Octree<T>::traverseTree(const std::function<void(const Node &)> &visit) const {
        std::function<void(const std::unique_ptr<Node> &)> traverse;
        traverse = [&](const std::unique_ptr<Node> &node) {
            if (!node)
                return;
            visit(*node);
            for (const auto &child: node->children) {
                traverse(child);
            }
        };
        traverse(root_);
    }

    template<typename T>
    bool Octree<T>::isExplored(const Eigen::Matrix<T, 3, 1> &point) const noexcept {
        return explored_points_.find(point) != explored_points_.end();
    }

    template<typename T>
    T Octree<T>::computeSimilarityGradient(
            const Node &node, const Image<> &target,
            const std::shared_ptr<processing::image::ImageComparator> &comparator) const {
        if (node.points.size() < 2)
            return 0;

        T max_similarity = std::numeric_limits<T>::lowest();
        T min_similarity = std::numeric_limits<T>::max();

        for (const auto &point: node.points) {
            T similarity = comparator->compare(target, Image<T>::fromViewPoint(point));
            max_similarity = std::max(max_similarity, similarity);
            min_similarity = std::min(min_similarity, similarity);
        }

        return max_similarity - min_similarity;
    }

    template<typename T>
    T Octree<T>::getBestScore() const {
        T best_score = std::numeric_limits<T>::lowest();
        traverseTree([&](const Node &node) {
            for (const auto &point: node.points) {
                best_score = std::max(best_score, point.getScore());
            }
        });
        return best_score;
    }

    template<typename T>
    void Octree<T>::failureRecovery() {
        T current_best_score = getBestScore();
        if (current_best_score > best_score_) {
            best_score_ = current_best_score;
            stuck_count_ = 0;
        } else {
            stuck_count_++;
        }

        if (stuck_count_ > max_stuck_iterations_) {
            LOG_INFO("Search stuck. Initiating failure recovery.");
            exploreDiscardedRegions();
            stuck_count_ = 0;
        }
    }

    template<typename T>
    void Octree<T>::exploreDiscardedRegions() {
        std::vector<ViewPoint<T>> discarded_points;
        traverseTree([&](const Node &node) {
            if (node.explored && node.points.empty()) {
                ViewPoint<T> new_point(node.center);
                discarded_points.push_back(new_point);
            }
        });

        for (const auto &point: discarded_points) {
            insert(point);
        }
    }

    template<typename T>
    bool Octree<T>::checkConvergence() const {
        T avg_score = 0;
        int count = 0;
        traverseTree([&](const Node &node) {
            for (const auto &point: node.points) {
                avg_score += point.getScore();
                count++;
            }
        });

        avg_score /= count;
        LOG_DEBUG("Average score: {}", avg_score);
        return avg_score >= best_score_ - 1e-3; // Convergence threshold
    }

} // namespace viewpoint

#endif // VIEWPOINT_OCTREE_HPP
