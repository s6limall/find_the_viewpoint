// File: filtering/heuristics/heuristic.cpp

#include "filtering/heuristics/heuristic.hpp"

namespace filtering::heuristics {
    std::vector<std::vector<double> > Heuristic::filter(const std::vector<std::vector<double> > &points,
                                                        double threshold) const {
        std::vector<std::vector<double> > filtered_points;
        for (const auto &point: points) {
            if (evaluate(point) >= threshold) {
                filtered_points.push_back(point);
            }
        }
        return filtered_points;
    }
}
