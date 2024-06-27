// File: filtering/heuristic.hpp

#ifndef HEURISTIC_FILTER_HPP
#define HEURISTIC_FILTER_HPP

#include "filtering/filter.hpp"
#include "heuristics/heuristic.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace filtering {
    class HeuristicFilter : public Filter {
    public:
        // Add a heuristic with an associated weight
        void addHeuristic(std::shared_ptr<heuristics::Heuristic> heuristic, double weight);

        // Evaluate a viewpoint using combined heuristics
        [[nodiscard]] double evaluate(const std::vector<double> &viewpoint) const;

        // Filter points based on a threshold
        [[nodiscard]] std::vector<std::vector<double> > filter(const std::vector<std::vector<double> > &points,
                                                               double threshold) const override;

        [[nodiscard]] std::string getName() const override;

    private:
        std::vector<std::pair<std::shared_ptr<heuristics::Heuristic>, double> > heuristics_;
    };
}

#endif // HEURISTIC_FILTER_HPP
