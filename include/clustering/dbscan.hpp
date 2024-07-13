// File: clustering/dbscan.hpp

#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include <vector>
#include <functional>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>
#include <execution>
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

        // void cluster(std::vector<ViewPoint<T> > &points);

        std::vector<Cluster<T> > cluster(std::vector<ViewPoint<T> > &points);

        const Cluster<T> &getBestCluster() const;

    private:
        mutable double epsilon_;
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
            metric_ = [this](const ViewPoint<T> &a, const ViewPoint<T> &b) { return defaultMetric(a, b); };
        }
    }


    template<typename T>
    std::vector<Cluster<T> > DBSCAN<T>::cluster(std::vector<ViewPoint<T> > &points) {
        if (epsilon_ < 0) {
            epsilon_ = estimateEpsilon(points);
            LOG_INFO("Estimated epsilon: {}", epsilon_);
        }

        int cluster_id = 0;
        clusters_.clear();
        best_cluster_ = {};
        best_cluster_.average_score = std::numeric_limits<double>::lowest();

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
            clusters_[i].cluster_id = static_cast<int>(i);
        }

        for (const auto &point: points) {
            if (point.getClusterId() != NOISE) {
                clusters_[point.getClusterId()].addPoint(point);
            }
        }

        for (auto &cluster: clusters_) {
            cluster.calculateAverageScore();
            if (cluster.getAverageScore() > best_cluster_.getAverageScore()) {
                best_cluster_ = cluster;
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
                for (int neighbor_index: neighbors) {
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

    /*template<typename T>
    double DBSCAN<T>::estimateEpsilon(const std::vector<ViewPoint<T> > &points) const {
        const size_t k = min_points_ - 1; // k-distance
        std::vector<double> k_distances(points.size());

        // Lambda function to calculate k-distance for each point
        auto compute_k_distance = [&](const ViewPoint<T> &point) {
            std::vector<double> neighbor_distances;
            neighbor_distances.reserve(points.size() - 1);

            // Compute distances to all other points
            for (const auto &other: points) {
                if (&point != &other) {
                    neighbor_distances.push_back(metric_(point, other));
                }
            }

            // Find the k-th nearest neighbor distance
            std::nth_element(neighbor_distances.begin(), neighbor_distances.begin() + k, neighbor_distances.end());
            return neighbor_distances[k];
        };

        // Compute k-distances
        std::transform(points.begin(), points.end(), k_distances.begin(), compute_k_distance);

        // Sort k-distances
        std::sort(k_distances.begin(), k_distances.end());

        // Determine the knee point using the maximum curvature method
        Eigen::VectorXd distances = Eigen::Map<Eigen::VectorXd>(k_distances.data(), k_distances.size());
        Eigen::VectorXd indices = Eigen::VectorXd::LinSpaced(k_distances.size(), 0, k_distances.size() - 1);

        // Compute the second derivative of the distances
        Eigen::VectorXd diff = distances.tail(distances.size() - 1) - distances.head(distances.size() - 1);
        Eigen::VectorXd second_diff = diff.tail(diff.size() - 1) - diff.head(diff.size() - 1);
        second_diff = second_diff.array().abs();

        const size_t knee_index = std::distance(second_diff.data(),
                                                std::max_element(second_diff.data(),
                                                                 second_diff.data() + second_diff.size()));

        return k_distances[knee_index];
    }*/


    template<typename T>
    const Cluster<T> &DBSCAN<T>::getBestCluster() const {
        return best_cluster_;
    }

}

#endif // DBSCAN_HPP


/*template<typename T>
    double DBSCAN<T>::estimateEpsilon(const std::vector<ViewPoint<T> > &points) const {
        std::vector<double> distances;
        distances.reserve(points.size() * (points.size() - 1) / 2);

        for (size_t i = 0; i < points.size(); ++i) {
            std::vector<double> neighbor_distances;
            for (size_t j = 0; j < points.size(); ++j) {
                if (i != j) {
                    neighbor_distances.push_back(metric_(points[i], points[j]));
                }
            }
            std::sort(neighbor_distances.begin(), neighbor_distances.end());
            distances.push_back(neighbor_distances[min_points_ - 1]);
        }

        std::sort(distances.begin(), distances.end());
        size_t knee_index = distances.size() / 2; // Simple heuristic, can be improved
        return distances[knee_index];
    }*/

/*template<typename T>
void DBSCAN<T>::cluster(std::vector<ViewPoint<T> > &points) {
    if (epsilon_ < 0) {
        epsilon_ = estimateEpsilon(points);
        LOG_INFO("Estimated epsilon: {}", epsilon_);
    }

    int cluster_id = 0;

    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].getClusterId() == UNCLASSIFIED) {
            const auto neighbors = regionQuery(points, static_cast<int>(i));
            if (neighbors.size() < min_points_) {
                points[i].setClusterId(NOISE);
            } else {
                expandCluster(points, static_cast<int>(i), cluster_id++);
            }
        }
    }
}*/

/*template<typename T>
  void DBSCAN<T>::expandCluster(std::vector<ViewPoint<T> > &points, int point_index, int cluster_id) const {
      std::queue<int> seeds;
      seeds.push(point_index);
      points[point_index].setClusterId(cluster_id);

      while (!seeds.empty()) {
          int current_point = seeds.front();
          seeds.pop();

          const auto neighbors = regionQuery(points, current_point);
          if (neighbors.size() >= min_points_) {
              for (int neighbor_index: neighbors) {
                  if (points[neighbor_index].getClusterId() == UNCLASSIFIED || points[neighbor_index].getClusterId()
                      == NOISE) {
                      if (points[neighbor_index].getClusterId() == UNCLASSIFIED) {
                          seeds.push(neighbor_index);
                      }
                      points[neighbor_index].setClusterId(cluster_id);
                  }
              }
          }
      }
  }*/


/*template<typename T>
std::vector<int> DBSCAN<T>::regionQuery(const std::vector<ViewPoint<T> > &points, int point_index) const {
    std::vector<int> neighbors;
    for (size_t i = 0; i < points.size(); ++i) {
        if (metric_(points[point_index], points[i]) < epsilon_) {
            neighbors.push_back(static_cast<int>(i));
        }
    }
    return neighbors;
}*/

/*template<typename T>
    double DBSCAN<T>::estimateEpsilon(const std::vector<ViewPoint<T> > &points) const {
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
    */
