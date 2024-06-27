// File: sampling/halton_sampler.cpp

#include "sampling/halton_sampler.hpp"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>

namespace sampling {
    std::vector<std::vector<double> > HaltonSampler::generate(
            int num_samples,
            const std::vector<double> &lower_bounds,
            const std::vector<double> &upper_bounds,
            bool adaptive_mode) {
        if (lower_bounds.size() != upper_bounds.size()) {
            throw std::invalid_argument("Lower and upper bounds must have the same dimension.");
        }

        int dimension = lower_bounds.size();
        if (dimension > 50) {
            throw std::invalid_argument("Dimension exceeds the number of available prime bases.");
        }

        std::vector<std::vector<double> > samples(num_samples, std::vector<double>(dimension));

        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < dimension; ++j) {
                double halton_value = haltonSequence(i + 1, prime_bases_[j]);
                samples[i][j] = lower_bounds[j] + halton_value * (upper_bounds[j] - lower_bounds[j]);
            }
        }

        if (adaptive_mode) {
            adapt(samples, lower_bounds, upper_bounds);
        }

        return samples;
    }

    void HaltonSampler::adapt(
            std::vector<std::vector<double> > &samples,
            const std::vector<double> &lower_bounds,
            const std::vector<double> &upper_bounds) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-0.01, 0.01);

        for (auto &sample: samples) {
            for (size_t j = 0; j < sample.size(); ++j) {
                double perturbation = distribution(generator);
                sample[j] = std::clamp(sample[j] + perturbation * (upper_bounds[j] - lower_bounds[j]), lower_bounds[j],
                                       upper_bounds[j]);
            }
        }
    }

    double HaltonSampler::haltonSequence(int index, int base) const {
        double result = 0.0;
        double f = 1.0;
        int i = index;

        while (i > 0) {
            f /= base;
            result += f * (i % base);
            i /= base;
        }

        return result;
    }
}
