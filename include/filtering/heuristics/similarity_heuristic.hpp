// File: filtering/heuristics/similarity_heuristic.hpp

#ifndef SIMILARITY_HEURISTIC_HPP
#define SIMILARITY_HEURISTIC_HPP

#include "filtering/heuristics/heuristic.hpp"
#include <vector>

namespace filtering::heuristics {
    class SimilarityHeuristic : public Heuristic {
    public:
        explicit SimilarityHeuristic(const std::vector<std::vector<double> > &target_distribution);

        double evaluate(const std::vector<double> &viewpoint) const override;

    private:
        std::vector<std::vector<double> > target_distribution_;
    };
}

#endif // SIMILARITY_HEURISTIC_HPP
