// File: sampling/constrained_spherical_sampler.cpp

#include "sampling/constrained_spherical_sampler.hpp"

// TODO: Generalize for any number of dimensions (currently works with only 3D)
namespace sampling {
    ConstrainedSphericalSampler::ConstrainedSphericalSampler(const double radius, const double tolerance) :
        radius_(radius), tolerance_(tolerance) {
        if (tolerance_ >= 1.0 || tolerance_ <= 0.0) {
            throw std::invalid_argument("Tolerance must be between 0 and 1.");
        }

        LOG_INFO("Creating ConstrainedSphericalSampler with radius: {} and tolerance: {}.", radius, tolerance);

        inner_radius_ = radius_ * (1.0 - tolerance_);
        outer_radius_ = radius_ * (1.0 + tolerance_);
        LOG_DEBUG("Inner Radius: {}, Outer Radius: {})", inner_radius_, outer_radius_);
    }

    std::pair<std::vector<double>, std::vector<double> > ConstrainedSphericalSampler::calculateBounds(
            const int dimensions) const {
        // Use the outer radius to define the bounding cube
        LOG_TRACE("Using outer radius {} to define the bounding cube.", outer_radius_);
        std::vector<double> lower_bounds(dimensions, -outer_radius_);
        std::vector<double> upper_bounds(dimensions, outer_radius_);
        LOG_DEBUG("Bounds calculated: lower_bounds = {}, upper_bounds = {}", lower_bounds, upper_bounds);
        return {lower_bounds, upper_bounds};
    }


    std::vector<std::vector<double> >
    ConstrainedSphericalSampler::generate(const int num_samples, const int dimensions) {
        if (dimensions != 3) {
            LOG_ERROR("Received dimensions: {}. ConstrainedSphericalSampler only supports dimensions = 3.", dimensions);
            throw std::invalid_argument("ConstrainedSphericalSampler only supports dimensions = 3.");
        }

        auto [lower_bounds, upper_bounds] = calculateBounds(dimensions);

        halton_sampler_.setAdaptive(true, [this](std::vector<double> &sample) {
            adaptSample(sample);
        });

        LOG_INFO("Generating {} samples using Halton sequences for low discrepancy.", num_samples);
        std::vector<std::vector<double> > samples = halton_sampler_.generate(num_samples, lower_bounds, upper_bounds);

        LOG_INFO("{} valid samples generated within spherical shell (Inner radius = {}, Outer radius = {})!",
                 samples.size(), inner_radius_, outer_radius_);
        return samples;
    }

    // TODO: Why use UNIT sphere? Use lower bound to upper bound? What if the bounds are far apart?
    // Normalize the sample to lie on the unit sphere
    void ConstrainedSphericalSampler::adaptSample(std::vector<double> &sample) const {
        std::vector<double> original_sample = sample;

        // Compute the length of the vector = sqrt(squared sum of elements)
        const double length = std::sqrt(std::inner_product(sample.begin(), sample.end(), sample.begin(), 0.0));
        LOG_TRACE("Normalizing sample {} to unit length: {}", original_sample, length);

        // Normalize the vector by dividing each element by the length
        for (auto &value: sample) {
            value /= length;
        }

        LOG_TRACE("Sample normalized to unit length: {}", sample);

        // Scale the sample to lie within the spherical shell
        const double radius = generateRandomRadius(sample);

        LOG_TRACE("Scaling sample {} to radius: {}", original_sample, radius);
        for (auto &value: sample) {
            value *= radius;
        }

        LOG_DEBUG("{}\t->\t{}", original_sample, sample);
    }

    bool ConstrainedSphericalSampler::isWithinShell(const std::vector<double> &point) const {
        const double distance = std::sqrt(std::inner_product(point.begin(), point.end(), point.begin(), 0.0));
        bool within_shell = distance >= inner_radius_ && distance <= outer_radius_;
        LOG_DEBUG("Checking if point {} is within the spherical shell: {}", point, within_shell);
        return within_shell;
    }

    double ConstrainedSphericalSampler::generateRandomRadius(std::vector<double> &sample) const {
        std::random_device rd;
        std::mt19937 gen(rd());
        // std::seed_seq seed(sample.begin(), sample.end()); // Forcing a dependence on the specific sample
        // std::mt19937 gen(seed);
        std::uniform_real_distribution dist(inner_radius_, outer_radius_);
        return dist(gen);
    }
}
