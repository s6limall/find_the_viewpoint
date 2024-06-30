// File: sampling/constrained_spherical_sampler.hpp

#ifndef CONSTRAINED_SPHERICAL_SAMPLER_HPP
#define CONSTRAINED_SPHERICAL_SAMPLER_HPP

#include "sampling/halton_sampler.hpp"
#include "common/logging/logger.hpp"

namespace sampling {

    class ConstrainedSphericalSampler {
    public:
        ConstrainedSphericalSampler(double radius, double tolerance);

        [[nodiscard]] std::vector<std::vector<double> > generate(int num_samples, int dimensions);

    private:
        double radius_;
        double tolerance_;
        double inner_radius_;
        double outer_radius_;
        HaltonSampler halton_sampler_;

        // Sets the lower and upper bounds based on the radius and dimensions
        [[nodiscard]] std::pair<std::vector<double>, std::vector<double> > calculateBounds(int dimensions) const;

        // Transforms unit cube Halton samples to fit within the spherical shell
        void adaptSample(std::vector<double> &sample) const;

        // Checks if a point lies within the spherical shell
        [[nodiscard]] bool isWithinShell(const std::vector<double> &point) const;

        // Generate a random radius in range [inner_radius, outer_radius]
        [[nodiscard]] double generateRandomRadius(std::vector<double> &sample) const;
    };

}

#endif // CONSTRAINED_SPHERICAL_SAMPLER_HPP
