// File: filtering/filter.cpp

#include "filtering/filter.hpp"

namespace filtering {

    std::vector<std::vector<double>> Filter::applyFilters(const std::vector<std::vector<double>>& points, double threshold) const {
        spdlog::debug("Applying filter chain with threshold: {}", threshold);
        std::vector<std::vector<double>> filtered_points = points;

        for (const auto& filter : filters_) {
            spdlog::debug("Applying filter: {}", filter->getName());
            filtered_points = filter->filter(filtered_points, threshold);
            spdlog::debug("Points remaining after filter '{}': {}", filter->getName(), filtered_points.size());
        }

        return filtered_points;
    }


}


