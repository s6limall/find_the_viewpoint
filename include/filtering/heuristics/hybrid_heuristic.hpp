//
// Created by ayush on 6/24/24.
//

#ifndef HYBRID_HEURISTIC_HPP
#define HYBRID_HEURISTIC_HPP

#endif //HYBRID_HEURISTIC_HPP

// File: filtering/heuristics/adaptive_combined_heuristic.hpp

#ifndef ADAPTIVE_COMBINED_HEURISTIC_HPP
#define ADAPTIVE_COMBINED_HEURISTIC_HPP

#include "heuristic_filter.hpp"
#include <vector>
#include <memory>

namespace filtering {
    namespace heuristics {
        class AdaptiveCombinedHeuristic : public HeuristicFilter {
        public:
            AdaptiveCombinedHeuristic();

            void addHeuristic(std::shared_ptr<HeuristicFilter> heuristic, double weight);

            double evaluate(const std::vector<double>& viewpoint) const override;

            void updateWeights(const std::vector<double>& feedback);

        private:
            std::vector<std::pair<std::shared_ptr<HeuristicFilter>, double>> heuristics_;
        };
    }
}

#endif // ADAPTIVE_COMBINED_HEURISTIC_HPP
