// File: viewpoint_optimizer.hpp

#ifndef VIEWPOINT_OPTIMIZER_HPP
#define VIEWPOINT_OPTIMIZER_HPP

#include <vector>
#include <functional>
#include <libcmaes/cmaes.h>
#include "types/viewpoint.hpp"
#include "types/image.hpp"
#include "viewpoint/evaluator.hpp"

class ViewpointOptimizer {
public:
    using EvaluationFunction = std::function<double(const ViewPoint<double>&)>;

    explicit ViewpointOptimizer(EvaluationFunction eval_func)
        : evaluation_function_(std::move(eval_func)) {}

    ViewPoint<double> optimize(const std::vector<double>& initial_guess, const std::vector<double>& lower_bounds, const std::vector<double>& upper_bounds, size_t max_iterations = 1000) {
        libcmaes::FitFunc fitness = [this](const double* x, const int N) {
            std::vector<double> point(x, x + N);
            ViewPoint<double> viewpoint(point[0], point[1], point[2]);
            return evaluation_function_(viewpoint);
        };

        libcmaes::CMAParameters<> parameters(initial_guess.data(), 0.5, initial_guess.size());
        parameters.set_bounds(lower_bounds, upper_bounds);
        parameters.set_max_iter(max_iterations);

        libcmaes::CMASolutions solutions = libcmaes::cmaes<>(fitness, parameters);
        std::vector<double> best_point = solutions.best_candidate().x;

        return ViewPoint<double>(best_point[0], best_point[1], best_point[2], solutions.best_candidate().get_fvalue());
    }

private:
    EvaluationFunction evaluation_function_;
};

#endif // VIEWPOINT_OPTIMIZER_HPP
