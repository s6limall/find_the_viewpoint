// File: sampling/constrained_spherical_sampler.hpp

#ifndef CONSTRAINED_SPHERICAL_SAMPLER_HPP
#define CONSTRAINED_SPHERICAL_SAMPLER_HPP

#include "sampling/sampler.hpp"
#include "sampling/halton_sampler.hpp"
#include <vector>
#include <random>
#include <functional>
#include <algorithm>

namespace sampling {

    class ConstrainedSphericalSampler : public Sampler {
    public:
        ConstrainedSphericalSampler(double inner_radius, double outer_radius);

        std::vector<std::vector<double> > generate(int num_samples, const std::vector<double> &lower_bounds,
                                                   const std::vector<double> &upper_bounds, bool use_halton) override;

        void adapt(std::vector<std::vector<double> > &samples, const std::vector<double> &lower_bounds,
                   const std::vector<double> &upper_bounds) override;

    private:
        double inner_radius_;
        double outer_radius_;
    };

}

#endif // CONSTRAINED_SPHERICAL_SAMPLER_HPP
