// File: spatial/octree.hpp

#ifndef SPATIAL_OCTREE_HPP
#define SPATIAL_OCTREE_HPP

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <random>
#include <vector>

#include "types/viewpoint.hpp"

namespace spatial {

    template<FloatingPoint T = double>
    class Octree {
    public:
        struct Node {
            Eigen::Vector3<T> center;
            T size;
            std::vector<ViewPoint<T>> points;
            std::array<std::unique_ptr<Node>, 8> children;
            bool explored = false;
            T max_acquisition = std::numeric_limits<T>::lowest();

            Node(const Eigen::Vector3<T> &center, T size) : center(center), size(size) {}
            [[nodiscard]] bool isLeaf() const noexcept { return children[0] == nullptr; } // node without children
        };

        Octree(const Eigen::Vector3<T> &center, T size, T min_size) :
            root_(std::make_unique<Node>(center, size)), min_size_(min_size) {}

        void split(Node &node);
        [[nodiscard]] bool shouldSplit(const Node &node,
                                       const std::optional<ViewPoint<T>> &best_viewpoint) const noexcept;
        static bool isWithinNode(const Node &node, const Eigen::Vector3<T> &position);
        [[nodiscard]] const Node *getRoot() const { return root_.get(); }
        Node *getRoot() { return root_.get(); }
        T getMinSize() const { return min_size_; }

    private:
        std::unique_ptr<Node> root_;
        T min_size_;
    };

    template<FloatingPoint T>
    bool Octree<T>::shouldSplit(const Node &node, const std::optional<ViewPoint<T>>& best_viewpoint) const noexcept {
        return node.size > min_size_ * T(2) &&
               (node.points.size() > 10 ||
                (best_viewpoint && (node.center - best_viewpoint->getPosition()).norm() < node.size));
    }


    template<FloatingPoint T>
    void Octree<T>::split(Node &node) {
        T child_size = node.size / T(2);

        LOG_INFO("Splitting node at {} with size {}", node.center.transpose(), node.size);
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector3<T> offset((i & 1) ? child_size : -child_size,
                                     (i & 2) ? child_size : -child_size,
                                     (i & 4) ? child_size : -child_size);
            node.children[i] = std::make_unique<Node>(node.center + offset * T(0.5), child_size);
        }

        for (const auto &point : node.points) {
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

    template<FloatingPoint T>
    bool Octree<T>::isWithinNode(const Node &node, const Eigen::Vector3<T> &position) {
        return (position - node.center).cwiseAbs().maxCoeff() <= node.size / 2;
    }


} // namespace spatial

#endif // SPATIAL_OCTREE_HPP
