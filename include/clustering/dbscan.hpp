// File: clustering/dbscan.hpp

#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include <vector>
#include <functional>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric> // For std::iota

#include <Eigen/Dense>

#include "types/viewpoint.hpp"
#include "types/cluster.hpp"
#include "common/logging/logger.hpp"

template<typename T>
using MetricFunction = std::function<double(const ViewPoint<T> &, const ViewPoint<T> &)>;

namespace clustering {

    template<typename T>
    class DBSCAN {
        static_assert(std::is_arithmetic_v<T>, "DBSCAN template must be numeric");

    public:
        explicit DBSCAN(int min_points, MetricFunction<T> metric = nullptr, double epsilon = -1.0) noexcept;

        std::vector<Cluster<T> > cluster(std::vector<ViewPoint<T> > &points);

        const Cluster<T> &getBestCluster() const;

    private:
        double epsilon_;
        int min_points_;
        MetricFunction<T> metric_;
        static constexpr int UNCLASSIFIED = -1, NOISE = -2;
        std::vector<Cluster<T> > clusters_;
        Cluster<T> best_cluster_;

        [[nodiscard]] std::vector<int> regionQuery(const std::vector<ViewPoint<T> > &points, int point_index) const;

        void expandCluster(std::vector<ViewPoint<T> > &points, int point_index, int cluster_id);

        [[nodiscard]] static double defaultMetric(const ViewPoint<T> &a, const ViewPoint<T> &b) noexcept;

        double estimateEpsilon(const std::vector<ViewPoint<T> > &points) const;
    };

    template<typename T>
    DBSCAN<T>::DBSCAN(const int min_points, MetricFunction<T> metric, const double epsilon) noexcept :
        epsilon_(epsilon), min_points_(min_points), metric_(std::move(metric)) {
        if (!metric_) {
            metric_ = [](const ViewPoint<T> &a, const ViewPoint<T> &b) { return defaultMetric(a, b); };
        }
    }

    template<typename T>
    std::vector<Cluster<T> > DBSCAN<T>::cluster(std::vector<ViewPoint<T> > &points) {
        if (points.empty()) {
            LOG_ERROR("Input points vector is empty");
            return {};
        }

        if (epsilon_ < 0) {
            epsilon_ = estimateEpsilon(points);
            LOG_INFO("Estimated epsilon: {}", epsilon_);
        }

        int cluster_id = 0;
        clusters_.clear();
        best_cluster_ = {};

        for (auto &point: points) {
            if (point.getClusterId() == UNCLASSIFIED) {
                const auto neighbors = regionQuery(points, &point - &points[0]);
                if (neighbors.size() < min_points_) {
                    point.setClusterId(NOISE);
                } else {
                    expandCluster(points, &point - &points[0], cluster_id++);
                }
            }
        }

        clusters_.resize(cluster_id);
        for (size_t i = 0; i < cluster_id; ++i) {
            clusters_[i].setClusterId(static_cast<int>(i));
        }

        for (const auto &point: points) {
            if (point.getClusterId() != NOISE) {
                clusters_[point.getClusterId()].addPoint(point);
            }
        }

        return clusters_;
    }

    template<typename T>
    std::vector<int> DBSCAN<T>::regionQuery(const std::vector<ViewPoint<T> > &points, int point_index) const {
        std::vector<int> neighbors;
        neighbors.reserve(points.size());
        const auto &point = points[point_index];

        for (const auto &other: points) {
            if (metric_(point, other) < epsilon_) {
                neighbors.push_back(&other - &points[0]);
            }
        }

        return neighbors;
    }

    template<typename T>
    void DBSCAN<T>::expandCluster(std::vector<ViewPoint<T> > &points, int point_index, int cluster_id) {
        std::queue<int> seeds;
        seeds.push(point_index);
        points[point_index].setClusterId(cluster_id);

        while (!seeds.empty()) {
            int current_point = seeds.front();
            seeds.pop();

            const auto neighbors = regionQuery(points, current_point);
            if (neighbors.size() >= min_points_) {
                for (const auto &neighbor_index: neighbors) {
                    auto &neighbor = points[neighbor_index];
                    if (neighbor.getClusterId() == UNCLASSIFIED || neighbor.getClusterId() == NOISE) {
                        if (neighbor.getClusterId() == UNCLASSIFIED) {
                            seeds.push(neighbor_index);
                        }
                        neighbor.setClusterId(cluster_id);
                    }
                }
            }
        }
    }

    template<typename T>
    double DBSCAN<T>::defaultMetric(const ViewPoint<T> &a, const ViewPoint<T> &b) noexcept {
        return (a.getPosition() - b.getPosition()).norm();
    }

    template<typename T>
    double DBSCAN<T>::estimateEpsilon(const std::vector<ViewPoint<T> > &points) const {
        if (points.size() < static_cast<size_t>(min_points_)) {
            LOG_ERROR("Insufficient points to estimate epsilon");
            throw std::runtime_error("Insufficient points to estimate epsilon");
        }

        std::vector<double> distances;
        distances.reserve(points.size() * (points.size() - 1) / 2);

        for (size_t i = 0; i < points.size(); ++i) {
            std::vector<double> neighbor_distances;
            for (size_t j = 0; j < points.size(); ++j) {
                if (i != j) {
                    neighbor_distances.push_back(metric_(points[i], points[j]));
                }
            }
            std::nth_element(neighbor_distances.begin(), neighbor_distances.begin() + min_points_ - 1,
                             neighbor_distances.end());
            distances.push_back(neighbor_distances[min_points_ - 1]);
        }

        std::sort(distances.begin(), distances.end());
        const size_t knee_index = distances.size() / 2; // Simple heuristic, can be improved
        return distances[knee_index];
    }

    template<typename T>
    const Cluster<T> &DBSCAN<T>::getBestCluster() const {
        return best_cluster_;
    }

} // namespace clustering

#endif // DBSCAN_HPP
