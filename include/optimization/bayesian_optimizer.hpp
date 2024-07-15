#ifndef BAYESIAN_OPTIMIZER_HPP
#define BAYESIAN_OPTIMIZER_HPP

#include <functional>
#include <algorithm>
#include <vector>
#include <Eigen/Core>
#include <random>

#include "sampling/sampler/halton_sampler.hpp"
#include "sampling/transformer/spherical_transformer.hpp"

template<typename GP, typename AF, typename T>
class BayesianOptimizer {
public:
    BayesianOptimizer(GP &gp, AF &af, std::function<double(const T &)> obj_func) :
        gp_(gp), af_(af), obj_func_(obj_func) {
    }

    T optimize(int iterations, double search_space_range) {
        // Halton sampler for generating initial points
        sampling::HaltonSampler halton_sampler(std::nullopt);
        halton_sampler.setAdaptive(true);
        std::vector<double> lower_bounds = {0.0, 0.0, 0.0};
        std::vector<double> upper_bounds = {1.0, 1.0, 1.0};

        // Spherical transformer to map samples to the search space
        sampling::SphericalShellTransformer transformer(0.1, search_space_range);

        for (int i = 0; i < iterations; ++i) {
            T best_point;
            double best_value = -std::numeric_limits<double>::infinity();

            auto search_space = halton_sampler.generate(100, lower_bounds, upper_bounds);
            for (const auto &sample: search_space) {
                std::vector<double> transformed_sample = transformer.transform(sample);
                T point = Eigen::Map<T>(transformed_sample.data());

                double value = af_(gp_, point);
                if (value > best_value) {
                    best_value = value;
                    best_point = point;
                }
            }

            double obj_value = obj_func_(best_point);
            gp_.addSample(best_point, obj_value);
            std::cout << "Iteration " << i << ": Point = " << best_point.transpose() << ", Value = " << obj_value <<
                    std::endl;
        }

        const auto &points = gp_.getPoints();
        const auto &values = gp_.getValues();
        auto max_iter = std::max_element(values.begin(), values.end());
        return points[std::distance(values.begin(), max_iter)];
    }

private:
    GP &gp_;
    AF &af_;
    std::function<double(const T &)> obj_func_;
};

#endif // BAYESIAN_OPTIMIZER_HPP
