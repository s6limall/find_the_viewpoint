// File: sampling/sampler.hpp

#ifndef SAMPLING_SAMPLER_HPP
#define SAMPLING_SAMPLER_HPP

#include <vector>

namespace sampling {
    class Sampler {
    public:
        virtual ~Sampler() = default;

        // Generates a set of sample points in the given bounds.
        // Adaptive mode allows dynamic/intelligent sampling.
        virtual std::vector<std::vector<double> > generate(
                int num_samples,
                const std::vector<double> &lower_bounds,
                const std::vector<double> &upper_bounds,
                bool adaptive_mode) = 0;

    protected:
        // Adaptively refines the samples.
        virtual void adapt(
                std::vector<std::vector<double> > &samples,
                const std::vector<double> &lower_bounds,
                const std::vector<double> &upper_bounds) = 0;
    };
}

#endif // SAMPLING_SAMPLER_HPP
