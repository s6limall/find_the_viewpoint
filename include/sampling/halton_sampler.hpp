// File: sampling/halton_sampler.hpp
// Generates a set of sample points in the given bounds.
// Adaptive mode allows dynamic/intelligent sampling.
// Dimensions inferred via number of elements in the bounds.
// For example, for 3D samples, lower_bounds = {0, 0, 0} and upper_bounds = {1, 1, 1}.
// For 6D samples, lower_bounds = {0, 0, 0, 0, 0, 0} and upper_bounds = {1, 1, 1, 1, 1, 1}.
// File: sampling/halton_sampler.hpp

#ifndef SAMPLING_HALTON_SAMPLER_HPP
#define SAMPLING_HALTON_SAMPLER_HPP

#include "sampling/sampler.hpp"
#include <vector>
#include <algorithm>

namespace sampling {
    class HaltonSampler : public Sampler {
    public:
        HaltonSampler() = default;

        // Generates a set of sample points in the given bounds.
        std::vector<std::vector<double> > generate(
                int num_samples,
                const std::vector<double> &lower_bounds,
                const std::vector<double> &upper_bounds,
                bool adaptive_mode) override;

    protected:
        // Refines the samples adaptively.
        void adapt(
                std::vector<std::vector<double> > &samples,
                const std::vector<double> &lower_bounds,
                const std::vector<double> &upper_bounds) override;

    private:
        // First 50 prime numbers for Halton sequence.
        static constexpr int prime_bases_[50] = {
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                179, 181, 191, 193, 197, 199, 211, 223, 227, 229
        };

        // Generates a Halton sequence value.
        [[nodiscard]] double haltonSequence(int index, int base) const;
    };
}

#endif // SAMPLING_HALTON_SAMPLER_HPP

