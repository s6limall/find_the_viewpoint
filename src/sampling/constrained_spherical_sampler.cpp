// File: sampling/constrained_spherical_sampler.cpp

#include "sampling/constrained_spherical_sampler.hpp"
#include <cmath>
#include <spdlog/spdlog.h>

namespace sampling {

    ConstrainedSphericalSampler::ConstrainedSphericalSampler(double inner_radius, double outer_radius)
        : inner_radius_(inner_radius), outer_radius_(outer_radius) {
        spdlog::info("ConstrainedSphericalSampler created with inner_radius: {} and outer_radius: {}", inner_radius_, outer_radius_);
    }

    std::vector<std::vector<double>> ConstrainedSphericalSampler::generate(int num_samples, const std::vector<double>& lower_bounds, const std::vector<double>& upper_bounds, bool use_halton) {
        std::vector<std::vector<double>> samples;
        HaltonSampler halton_sampler;

        // Generate initial samples using Halton sequence
        samples = halton_sampler.generate(num_samples, lower_bounds, upper_bounds, use_halton);

        // Adapt samples to be within the spherical shell
        adapt(samples, lower_bounds, upper_bounds);

        return samples;
    }

    void ConstrainedSphericalSampler::adapt(std::vector<std::vector<double>>& samples, const std::vector<double>& lower_bounds, const std::vector<double>& upper_bounds) {
        std::uniform_real_distribution<double> radius_distribution(inner_radius_, outer_radius_);
        std::uniform_real_distribution<double> theta_distribution(0, 2 * M_PI);
        std::uniform_real_distribution<double> phi_distribution(0, M_PI);

        std::mt19937 rng(std::random_device{}());

        for (auto& sample : samples) {
            double r = radius_distribution(rng);
            double theta = theta_distribution(rng);
            double phi = phi_distribution(rng);

            double x = r * sin(phi) * cos(theta);
            double y = r * sin(phi) * sin(theta);
            double z = r * cos(phi);

            sample = {x, y, z};
        }
    }

}
