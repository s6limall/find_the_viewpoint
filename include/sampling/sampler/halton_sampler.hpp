// File: sampling/halton_sampler.hpp

// Generates a set of sample points in the given bounds.
// Dimensions inferred via number of elements in the bounds.
// For example, for 3D samples, lower_bounds = {0, 0, 0} and upper_bounds = {1, 1, 1}.
// For 6D samples, lower_bounds = {0, 0, 0, 0, 0, 0} and upper_bounds = {1, 1, 1, 1, 1, 1}.

#ifndef SAMPLING_HALTON_SAMPLER_HPP
#define SAMPLING_HALTON_SAMPLER_HPP

#include <cmath>
#include <array>
#include <vector>
#include <random>
#include <optional>
#include <algorithm>
#include <functional>

#include "sampling/sampler.hpp"
#include "common/logging/logger.hpp"

namespace sampling {

    class HaltonSampler final : public Sampler {
    public:
        using Sampler::Sampler;

        Points<double> generate(
                size_t num_samples,
                const std::vector<double> &lower_bounds,
                const std::vector<double> &upper_bounds) override;

        std::vector<double> next() override;

    protected:
        void adapt(std::vector<double> &sample) override;

    private:
        // First 50 prime numbers for Halton sequence bases.
        static constexpr std::array<int, 50> primes_ = {
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                179, 181, 191, 193, 197, 199, 211, 223, 227, 229
        };

        Points<double> untouched_samples_;

        // Computes the Halton sequence value for a given index and base.
        static double halton(int index, int base) noexcept;

        // Finds the next prime number greater than the given number.
        static int nextPrime(int after) noexcept;

        // Scales the sample based on the bounds and Halton sequence value.
        [[nodiscard]] std::vector<double> scale(size_t index) const;
    };

}

#endif // SAMPLING_HALTON_SAMPLER_HPP
