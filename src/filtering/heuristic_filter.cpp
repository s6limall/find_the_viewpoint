// File: filtering/heuristic.cpp

#include "filtering/heuristic_filter.hpp"

namespace filtering {
    void HeuristicFilter::addHeuristic(std::shared_ptr<heuristics::Heuristic> heuristic, double weight) {
        heuristics_.emplace_back(heuristic, weight);
    }

    std::string HeuristicFilter::getName() const {
        return "Heuristic Filter";
    }

    double HeuristicFilter::evaluate(const std::vector<double> &viewpoint) const {
        double total_score = 0.0;
        double total_weight = 0.0;

        for (const auto &[heuristic, weight]: heuristics_) {
            double score = heuristic->evaluate(viewpoint);
            spdlog::debug("Heuristic {} score: {}", typeid(*heuristic).name(), score);
            total_score += score * weight;
            total_weight += weight;
        }

        return total_weight == 0 ? 0 : total_score / total_weight;
    }

    std::vector<std::vector<double> > HeuristicFilter::filter(const std::vector<std::vector<double> > &points,
                                                              double threshold) const {
        spdlog::info("Filtering points with threshold {}", threshold);
        std::vector<std::vector<double> > filtered_points;

        for (const auto &point: points) {
            double score = evaluate(point);
            spdlog::debug("Point score: {}", score);
            if (score >= threshold) {
                filtered_points.push_back(point);
            }
        }

        spdlog::info("Filtered points count: {}", filtered_points.size());
        return filtered_points;
    }
}
