// File: filtering/heuristics/heuristic.hpp

#ifndef HEURISTIC_HPP
#define HEURISTIC_HPP

#include <vector>
#include <memory>

namespace filtering::heuristics {
    class Heuristic {
    public:
        virtual ~Heuristic() = default;

        // Evaluate a viewpoint using the heuristic criteria
        virtual double evaluate(const std::vector<double> &viewpoint) const = 0;

        // Filter viewpoints based on a threshold
        virtual std::vector<std::vector<double> > filter(const std::vector<std::vector<double> > &points,
                                                         double threshold) const;

    protected:
        Heuristic() = default;
    };
}

#endif // HEURISTIC_HPP
