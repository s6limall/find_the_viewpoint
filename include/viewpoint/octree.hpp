// File: viewpoint/octree.hpp

#ifndef OCTREE_HPP
#define OCTREE_HPP

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <vector>
#include "common/logging/logger.hpp"
#include "types/viewpoint.hpp"

namespace viewpoint {

    template<typename T = double>
    class Octree {
    public:
        struct Node {
            Eigen::Matrix<T, 3, 1> center;
            T size;
            std::vector<ViewPoint<T>> points;
            std::array<std::unique_ptr<Node>, 8> children;

            Node(const Eigen::Matrix<T, 3, 1> &center, T size) noexcept :
                center(center), size(size), points(), children{} {}
        };

        Octree(T resolution, size_t max_points_per_node = 10) noexcept;

        void insert(const ViewPoint<T> &point);
        [[nodiscard]] std::vector<ViewPoint<T>> search(const Eigen::Matrix<T, 3, 1> &center, T radius) const;

    private:
        std::unique_ptr<Node> root_;
        T resolution_;
        size_t max_points_per_node_;

        void insert(std::unique_ptr<Node> &node, const ViewPoint<T> &point);
        [[nodiscard]] bool isWithinBounds(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept;
        [[nodiscard]] size_t getOctant(const Node &node, const Eigen::Matrix<T, 3, 1> &point) const noexcept;
    };

    template<typename T>
    Octree<T>::Octree(T resolution, size_t max_points_per_node) noexcept :
        resolution_(resolution), max_points_per_node_(max_points_per_node) {
        root_ = std::make_unique<Node>(Eigen::Matrix<T, 3, 1>::Zero(), resolution);
    }

    template<typename T>
    void Octree<T>::insert(const ViewPoint<T> &point) {
        insert(root_, point);
    }

    template<typename T>
    void Octree<T>::insert(std::unique_ptr<Node> &node, const ViewPoint<T> &point) {
        if (!isWithinBounds(*node, point.getPosition())) {
            LOG_WARN("Point ({}, {}, {}) is out of bounds for this node.", point.getPosition().x(),
                     point.getPosition().y(), point.getPosition().z());
            return;
        }

        if (node->points.size() < max_points_per_node_ || node->size <= resolution_) {
            node->points.push_back(point);
            LOG_INFO("Inserted point ({}, {}, {}) into the node.", point.getPosition().x(), point.getPosition().y(),
                     point.getPosition().z());
            return;
        }

        if (node->children[0] == nullptr) {
            for (size_t i = 0; i < 8; ++i) {
                Eigen::Matrix<T, 3, 1> new_center = node->center;
                T offset = node->size / 4;
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
                node->children[i] = std::make_unique<Node>(new_center, node->size / 2);
            }
        }

        size_t octant = getOctant(*node, point.getPosition());
        insert(node->children[octant], point);
    }

    template<typename T>
    std::vector<ViewPoint<T>> Octree<T>::search(const Eigen::Matrix<T, 3, 1> &center, T radius) const {
        std::vector<ViewPoint<T>> results;
        std::function<void(const std::unique_ptr<Node> &)> searchRecursive;
        searchRecursive = [&](const std::unique_ptr<Node> &node) {
            if (!node)
                return;
            if ((node->center - center).norm() > radius + node->size / 2)
                return;
            for (const auto &point: node->points) {
                if ((point.getPosition() - center).norm() <= radius) {
                    results.push_back(point);
                }
            }
            for (const auto &child: node->children) {
                searchRecursive(child);
            }
        };
        searchRecursive(root_);
        return results;
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

} // namespace viewpoint

#endif // OCTREE_HPP
