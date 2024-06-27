// File: filtering/heuristics/similarity_heuristic.cpp

#include "filtering/heuristics/similarity_heuristic.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace filtering::heuristics {
        SimilarityHeuristic::SimilarityHeuristic(const std::vector<std::vector<double>>& target_distribution)
            : target_distribution_(target_distribution) {}

        double SimilarityHeuristic::evaluate(const std::vector<double>& viewpoint) const {
            double min_distance = std::numeric_limits<double>::max();
            for (const auto& target : target_distribution_) {
                double distance = std::sqrt(std::inner_product(viewpoint.begin(), viewpoint.end(), target.begin(), 0.0,
                                                               std::plus<double>(),
                                                               [](double a, double b) { return (a - b) * (a - b); }));
                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
            return 1.0 / (1.0 + min_distance); // Higher score for closer match to any target point
        }
}
