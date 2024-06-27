/*
// File: sampling/lhs_sampler.hpp

#ifndef SAMPLING_LHS_SAMPLER_HPP
#define SAMPLING_LHS_SAMPLER_HPP

#include "sampling/sampler.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace sampling {
    class LHSSampler : public Sampler {
    public:
        explicit LHSSampler(int dimension, unsigned int seed = 1);

        std::vector<std::vector<double> > generate(int num_samples,
                                                   const std::vector<double> &lower_bounds,
                                                   const std::vector<double> &upper_bounds) override;

    private:
        int dimension_;
        std::mt19937 engine_;
    };
}

#endif // SAMPLING_LHS_SAMPLER_HPP
*/
