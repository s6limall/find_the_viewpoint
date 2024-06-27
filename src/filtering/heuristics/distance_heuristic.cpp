// File: filtering/heuristics/distance_heuristic.cpp

#include "filtering/heuristics/distance_heuristic.hpp"
#include <cmath>
#include <numeric>

namespace filtering::heuristics {
    DistanceHeuristic::DistanceHeuristic(const std::vector<double> &target) :
        target_(target) {
    }

    double DistanceHeuristic::evaluate(const std::vector<double> &viewpoint) const {
        double distance = std::sqrt(std::inner_product(viewpoint.begin(), viewpoint.end(), target_.begin(), 0.0,
                                                       std::plus<>(),
                                                       [](double a, double b) { return (a - b) * (a - b); }));
        return 1.0 / (1.0 + distance); // Higher score for closer points
    }
}
