// File: filtering/heuristics/distance_heuristic.hpp

#ifndef DISTANCE_HEURISTIC_HPP
#define DISTANCE_HEURISTIC_HPP

#include "filtering/heuristics/heuristic.hpp"
#include <vector>

namespace filtering::heuristics {
    class DistanceHeuristic : public Heuristic {
    public:
        explicit DistanceHeuristic(const std::vector<double> &target);

        double evaluate(const std::vector<double> &viewpoint) const override;

    private:
        std::vector<double> target_;
    };
}

#endif // DISTANCE_HEURISTIC_HPP
