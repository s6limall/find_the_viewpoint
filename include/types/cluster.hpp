// File: types/cluster.hpp

#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <optional>
#include <vector>
#include "types/viewpoint.hpp"

template<typename T = double>
class Cluster {
public:
    constexpr Cluster() noexcept = default;

    [[nodiscard]] constexpr int getClusterId() const noexcept { return cluster_id_; }
    [[nodiscard]] std::vector<ViewPoint<T>> &getPoints() noexcept { return points_; }
    [[nodiscard]] const std::vector<ViewPoint<T>> &getPoints() const noexcept { return points_; }
    [[nodiscard]] constexpr size_t size() const noexcept { return points_.size(); }
    [[nodiscard]] constexpr double getAverageScore() const noexcept { return average_score_; }

    [[nodiscard]] const ViewPoint<T> &getBest() const noexcept { return best_.value(); }

    constexpr void setClusterId(const int cluster_id) noexcept { cluster_id_ = cluster_id; }

    void addPoint(const ViewPoint<T> &point) {
        points_.push_back(point);
        total_score_ += point.getScore();
        average_score_ = total_score_ / points_.size();

        if (!best_ || point.getScore() > best_->getScore()) {
            best_ = point;
        }
    }

private:
    int cluster_id_ = -1; // -1 = unclassified, -2 = noise
    double average_score_ = 0.0;
    double total_score_ = 0.0;
    std::vector<ViewPoint<T>> points_;
    std::optional<ViewPoint<T>> best_ = std::nullopt;
};

#endif // CLUSTER_HPP
