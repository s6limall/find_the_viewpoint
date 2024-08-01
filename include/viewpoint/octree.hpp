#ifndef VIEWPOINT_OCTREE_HPP
#define VIEWPOINT_OCTREE_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
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
            root_(std::make_unique<Node>(origin, size)), iteration_(0), max_iterations_(1000) {
            LOG_DEBUG("Octree initialized with origin ({}, {}, {}), size {}, resolution {}, max_points_per_node {}",
                      origin.x(), origin.y(), origin.z(), size, resolution, max_points_per_node);
        }

        void insert(const ViewPoint<T> &point);
        void refine(int max_depth, const Image<> &target,
                    const std::shared_ptr<processing::image::ImageComparator> &comparator,
                    const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr);
        std::vector<ViewPoint<T>> sampleNewViewpoints(
                size_t n, const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) const;
        bool checkConvergence() const;
        void failureRecovery();
        bool isWithinBounds(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept;
        void traverseTree(const std::function<void(const Node &)> &visit) const;
        const Node *getRoot() const noexcept { return root_.get(); }
        T getBestScore() const;

    private:
        std::unique_ptr<Node> root_;
        T resolution_;
        size_t max_points_per_node_;
        std::vector<T> score_history_;
        size_t iteration_;
        size_t max_iterations_;

        void insert(Node *node, const ViewPoint<T> &point);
        void refineNode(Node &node, int depth, int max_depth, const Image<> &target,
                        const std::shared_ptr<processing::image::ImageComparator> &comparator,
                        const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr);
        size_t getOctant(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept;
        void subdivide(Node &node);
        T computeSimilarityGradient(const Node &node, const Image<> &target,
                                    const std::shared_ptr<processing::image::ImageComparator> &comparator) const;
        void exploreDiscardedRegions();
        bool shouldSplitNode(const Node &node, int depth) const;
        void updateNodeStatistics(
                Node &node, const Image<> &target,
                const std::shared_ptr<processing::image::ImageComparator> &comparator,
                const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr);
        T computeUCB(T mean, T variance, size_t n) const;
    };

    template<FloatingPoint T>
    void Octree<T>::insert(const ViewPoint<T> &point) {
        if (!isWithinBounds(*root_, point.getPosition())) {
            LOG_WARN("Point ({}, {}, {}) is out of bounds for this Octree.", point.getPosition().x(),
                     point.getPosition().y(), point.getPosition().z());
            return;
        }
        insert(root_.get(), point);
    }

    template<FloatingPoint T>
    void Octree<T>::insert(Node *node, const ViewPoint<T> &point) {
        while (true) {
            if (node->points.size() < max_points_per_node_ || node->size <= resolution_) {
                node->points.push_back(point);
                LOG_DEBUG("Inserted point at ({}, {}, {}) into node at ({}, {}, {}) with size {}",
                          point.getPosition().x(), point.getPosition().y(), point.getPosition().z(), node->center.x(),
                          node->center.y(), node->center.z(), node->size);
                return;
            }

            if (node->isLeaf()) {
                LOG_DEBUG("Subdividing node at ({}, {}, {}) with size {}", node->center.x(), node->center.y(),
                          node->center.z(), node->size);
                subdivide(*node);
            }

            size_t octant = getOctant(*node, point.getPosition());
            node = node->children[octant].get();
        }
    }

    template<FloatingPoint T>
    void
    Octree<T>::refine(int max_depth, const Image<> &target,
                      const std::shared_ptr<processing::image::ImageComparator> &comparator,
                      const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) {
        LOG_INFO("Starting refinement with max depth {}", max_depth);
        refineNode(*root_, 0, max_depth, target, comparator, gpr);
        iteration_++;
        T best_score = getBestScore();
        score_history_.push_back(best_score);
        LOG_INFO("Refinement complete for iteration {}: best score = {}", iteration_, best_score);
    }

    template<FloatingPoint T>
    void
    Octree<T>::refineNode(Node &node, int depth, int max_depth, const Image<> &target,
                          const std::shared_ptr<processing::image::ImageComparator> &comparator,
                          const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) {
        if (depth >= max_depth || node.explored) {
            LOG_DEBUG("Node at ({}, {}, {}) reached max depth or already explored", node.center.x(), node.center.y(),
                      node.center.z());
            return;
        }

        updateNodeStatistics(node, target, comparator, gpr);

        if (shouldSplitNode(node, depth)) {
            LOG_DEBUG("Splitting node at ({}, {}, {}) with size {}", node.center.x(), node.center.y(), node.center.z(),
                      node.size);
            subdivide(node);
            for (auto &child: node.children) {
                if (child) {
                    refineNode(*child, depth + 1, max_depth, target, comparator, gpr);
                }
            }
        }

        node.explored = true;
    }

    template<FloatingPoint T>
    std::vector<ViewPoint<T>> Octree<T>::sampleNewViewpoints(
            size_t n, const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
            const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) const {
        std::vector<ViewPoint<T>> new_viewpoints;
        std::priority_queue<std::pair<T, ViewPoint<T>>> pq;

        traverseTree([&](const Node &node) {
            for (const auto &point: node.points) {
                auto [mean, variance] = gpr.predict(point.getPosition());
                T ucb = computeUCB(mean, variance, node.points.size());
                pq.emplace(ucb, point);
                LOG_DEBUG("Predicted UCB for point at ({}, {}, {}) is {}", point.getPosition().x(),
                          point.getPosition().y(), point.getPosition().z(), ucb);
            }
        });

        new_viewpoints.reserve(std::min(n, pq.size()));
        for (size_t i = 0; i < n && !pq.empty(); ++i) {
            new_viewpoints.push_back(std::move(pq.top().second));
            pq.pop();
        }

        LOG_INFO("Sampled {} new viewpoints", new_viewpoints.size());
        return new_viewpoints;
    }

    template<FloatingPoint T>
    bool Octree<T>::isWithinBounds(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept {
        T half_size = node.size / 2;
        bool within_bounds = (point.array() >= (node.center.array() - half_size)).all() &&
                             (point.array() <= (node.center.array() + half_size)).all();
        LOG_DEBUG("Point ({}, {}, {}) is within bounds: {}", point.x(), point.y(), point.z(), within_bounds);
        return within_bounds;
    }

    template<FloatingPoint T>
    size_t Octree<T>::getOctant(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept {
        size_t octant = 0;
        if (point.x() >= node.center.x())
            octant |= 1;
        if (point.y() >= node.center.y())
            octant |= 2;
        if (point.z() >= node.center.z())
            octant |= 4;
        LOG_DEBUG("Point ({}, {}, {}) assigned to octant {}", point.x(), point.y(), point.z(), octant);
        return octant;
    }

    template<FloatingPoint T>
    void Octree<T>::subdivide(Node &node) {
        T new_size = node.size / 2;
        T offset = new_size / 2;

        for (size_t i = 0; i < 8; ++i) {
            Eigen::Matrix<T, 3, 1> new_center = node.center;
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
            node.children[i] = std::make_unique<Node>(new_center, new_size);
            LOG_DEBUG("Created child node at ({}, {}, {}) with size {}", new_center.x(), new_center.y(), new_center.z(),
                      new_size);
        }

        for (const auto &point: node.points) {
            size_t octant = getOctant(node, point.getPosition());
            node.children[octant]->points.push_back(point);
            LOG_DEBUG("Moved point ({}, {}, {}) to child octant {}", point.getPosition().x(), point.getPosition().y(),
                      point.getPosition().z(), octant);
        }
        node.points.clear();
    }

    template<FloatingPoint T>
    void Octree<T>::traverseTree(const std::function<void(const Node &)> &visit) const {
        std::function<void(const Node &)> traverse = [&](const Node &node) {
            visit(node);
            for (const auto &child: node.children) {
                if (child)
                    traverse(*child);
            }
        };
        traverse(*root_);
    }

    template<FloatingPoint T>
    T Octree<T>::computeSimilarityGradient(
            const Node &node, const Image<> &target,
            const std::shared_ptr<processing::image::ImageComparator> &comparator) const {
        if (node.points.size() < 2)
            return 0;

        T max_similarity = std::numeric_limits<T>::lowest();
        T min_similarity = std::numeric_limits<T>::max();

        for (const auto &point: node.points) {
            T similarity = comparator->compare(target, Image<T>::fromViewPoint(point));
            LOG_INFO("Computed similarity for point at ({}, {}, {}): {}", point.getPosition().x(),
                     point.getPosition().y(), point.getPosition().z(), similarity);
            max_similarity = std::max(max_similarity, similarity);
            min_similarity = std::min(min_similarity, similarity);
        }

        T gradient = max_similarity - min_similarity;
        LOG_DEBUG("Computed similarity gradient for node at ({}, {}, {}): {}", node.center.x(), node.center.y(),
                  node.center.z(), gradient);
        return gradient;
    }

    template<FloatingPoint T>
    T Octree<T>::getBestScore() const {
        T best_score = std::numeric_limits<T>::lowest();
        traverseTree([&](const Node &node) {
            for (const auto &point: node.points) {
                best_score = std::max(best_score, point.getScore());
            }
        });
        LOG_INFO("Best score found: {}", best_score);
        return best_score;
    }

    template<FloatingPoint T>
    void Octree<T>::failureRecovery() {
        LOG_WARN("Starting failure recovery.");
        exploreDiscardedRegions();
    }

    template<FloatingPoint T>
    void Octree<T>::exploreDiscardedRegions() {
        std::vector<ViewPoint<T>> discarded_points;
        traverseTree([&](const Node &node) {
            if (node.explored && node.points.empty()) {
                ViewPoint<T> new_point(node.center);
                discarded_points.push_back(new_point);
                LOG_DEBUG("Exploring discarded region at node ({}, {}, {})", node.center.x(), node.center.y(),
                          node.center.z());
            }
        });

        for (const auto &point: discarded_points) {
            insert(point);
        }
    }

    template<FloatingPoint T>
    bool Octree<T>::checkConvergence() const {
        const size_t window_size = 10;
        const T improvement_threshold = static_cast<T>(1e-4);

        if (score_history_.size() < window_size) {
            return false;
        }

        T avg_improvement = 0;
        for (size_t i = score_history_.size() - window_size; i < score_history_.size() - 1; ++i) {
            avg_improvement += (score_history_[i + 1] - score_history_[i]) / window_size;
        }

        bool converged = std::abs(avg_improvement) < improvement_threshold;
        LOG_INFO("Convergence check: {}", converged);
        return converged;
    }

    template<FloatingPoint T>
    bool Octree<T>::shouldSplitNode(const Node &node, int depth) const {
        const T density_threshold = static_cast<T>(0.7);
        const T gradient_threshold = static_cast<T>(0.1);
        const T size_threshold = resolution_ * 2;

        T point_density = static_cast<T>(node.points.size()) / std::pow(8, depth);

        bool should_split = (point_density > density_threshold || node.similarity_gradient > gradient_threshold) &&
                            node.size > size_threshold;
        LOG_DEBUG("Should split node at ({}, {}, {}) with size {}: {}", node.center.x(), node.center.y(),
                  node.center.z(), node.size, should_split);
        return should_split;
    }

    template<FloatingPoint T>
    void Octree<T>::updateNodeStatistics(
            Node &node, const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
            const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) {
        for (auto &point: node.points) {
            if (!point.hasScore()) {
                auto similarity_score =
                        comparator->compare(target.getImage(), Image<T>::fromViewPoint(point).getImage());
                point.setScore(similarity_score);
                LOG_DEBUG("Computed similarity score for point ({}, {}, {}): {}", point.getPosition().x(),
                          point.getPosition().y(), point.getPosition().z(), similarity_score);
            }
            if (!point.hasUncertainty()) {
                auto [mean, variance] = gpr.predict(point.getPosition());
                point.setUncertainty(variance);
                LOG_DEBUG("Computed uncertainty for point ({}, {}, {}): mean = {}, variance = {}",
                          point.getPosition().x(), point.getPosition().y(), point.getPosition().z(), mean, variance);
            }
        }
        node.similarity_gradient = computeSimilarityGradient(node, target, comparator);
    }

    template<FloatingPoint T>
    T Octree<T>::computeUCB(T mean, T variance, size_t n) const {
        T exploration_factor =
                std::max(static_cast<T>(0.1), 1 - static_cast<T>(iteration_) / static_cast<T>(max_iterations_));
        T ucb = mean + exploration_factor * std::sqrt(2 * std::log(static_cast<T>(n)) / (n + 1)) * std::sqrt(variance);
        LOG_DEBUG("Computed UCB: mean = {}, variance = {}, UCB = {}", mean, variance, ucb);
        return ucb;
    }

} // namespace viewpoint

#endif // VIEWPOINT_OCTREE_HPP
