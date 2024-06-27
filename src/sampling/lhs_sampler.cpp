/*// File: sampling/lhs_sampler.cpp

#include "sampling/lhs_sampler.hpp"
#include <cassert>
#include <cmath>
#include <numeric>

namespace sampling {

    LHSSampler::LHSSampler(int dimension, unsigned int seed) :
        dimension_(dimension), engine_(seed) {
        assert(dimension > 0 && "Dimension must be positive");
        spdlog::info("Initializing LHSSampler with dimension {}", dimension_);
    }

    std::vector<std::vector<double>> LHSSampler::generateSamples(int num_samples,
                                                                 const std::vector<double>& lower_bounds,
                                                                 const std::vector<double>& upper_bounds) {
        assert(num_samples > 0 && "Number of samples must be positive");
        assert(lower_bounds.size() == upper_bounds.size() && "Bounds size mismatch");
        assert(lower_bounds.size() == static_cast<size_t>(dimension_) && "Dimension mismatch with bounds size");

        spdlog::info("Generating {} samples in {} dimensions", num_samples, dimension_);
        std::vector<std::vector<double>> samples(num_samples, std::vector<double>(dimension_));

        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

        for (int dim = 0; dim < dimension_; ++dim) {
            std::vector<int> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), engine_);

            for (int i = 0; i < num_samples; ++i) {
                double perturbed_sample = (indices[i] + uniform_dist(engine_)) / num_samples;
                samples[i][dim] = lower_bounds[dim] + perturbed_sample * (upper_bounds[dim] - lower_bounds[dim]);
            }
        }

        for (int i = 0; i < num_samples; ++i) {
            spdlog::debug("Sample {}: {}", i, fmt::join(samples[i], ", "));
        }

        return samples;
    }

}*/