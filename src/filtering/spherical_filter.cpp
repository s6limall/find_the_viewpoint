// File: filtering/spheical_filter.cpp

#include "filtering/spherical_filter.hpp"

namespace filtering {

    SphericalFilter::SphericalFilter(double radius, double thickness_ratio) : radius_(radius) {
        if (radius_ <= 0 || thickness_ratio <= 0 || thickness_ratio >= 1) {
            spdlog::error("Invalid radius or thickness ratio. Radius: {}, Thickness Ratio: {}", radius, thickness_ratio);
            throw std::invalid_argument("Invalid radius or thickness ratio.");
        }
        calculateShellBounds(thickness_ratio);
        spdlog::info("SphericalFilter initialized with radius: {}, inner_radius: {}, outer_radius: {}", radius_, inner_radius_, outer_radius_);
    }

    void SphericalFilter::calculateShellBounds(double thickness_ratio) {
        inner_radius_ = radius_ * (1.0 - thickness_ratio);
        outer_radius_ = radius_ * (1.0 + thickness_ratio);
    }

    bool SphericalFilter::isWithinShell(const Eigen::Vector3f &point) const {
        double norm = point.norm();
        return norm >= inner_radius_ && norm <= outer_radius_;
    }

    std::vector<std::vector<double>> SphericalFilter::filter(const std::vector<std::vector<double>>& points, double threshold) const {
        spdlog::info("SphericalFilter: Filtering {} points", points.size());
        std::vector<std::vector<double>> filtered_points;
        for (const auto &sample : points) {
            Eigen::Vector3f point(sample[0], sample[1], sample[2]);
            if (isWithinShell(point)) {
                filtered_points.push_back(sample);
            }
        }
        spdlog::info("SphericalFilter: {} points remaining after filtering", filtered_points.size());
        return applyFilters(filtered_points, threshold); // Apply chained filters
    }

    std::string SphericalFilter::getName() const { return "SphericalFilter"; }

}