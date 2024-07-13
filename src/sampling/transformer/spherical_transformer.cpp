// File: sampling/spherical_transformer.cpp

#include "sampling/transformer/spherical_transformer.hpp"

namespace sampling {

    SphericalShellTransformer::SphericalShellTransformer(double inner_radius, double outer_radius) :
        inner_radius_(inner_radius), outer_radius_(outer_radius) {
        if (inner_radius <= 0 || outer_radius <= 0 || inner_radius >= outer_radius) {
            LOG_ERROR("Invalid radius values for spherical shell: inner_radius = {}, outer_radius = {}", inner_radius,
                      outer_radius);
            throw std::invalid_argument("Invalid radius values for spherical shell.");
        }
        LOG_INFO("SphericalShellTransformer initialized with inner_radius = {}, outer_radius = {}", inner_radius,
                 outer_radius);
    }

    std::vector<double> SphericalShellTransformer::transform(const std::vector<double> &sample) const {
        validate(sample);

        const auto [radius_factor, azimuth_factor, polar_factor] = std::tuple{sample[0], sample[1], sample[2]};
        const double radius = inner_radius_ + (outer_radius_ - inner_radius_) * radius_factor;
        const double azimuth = 2.0 * PI() * azimuth_factor;
        const double polar = std::acos(1.0 - 2.0 * polar_factor);

        const Eigen::Vector3d cartesian_coordinates = sphericalToCartesian(radius, azimuth, polar);

        LOG_TRACE("Transformed sample: (radius_factor, azimuth_factor, polar_factor) = ({}, {}, {}) to (x, y, z) = {}",
                  radius_factor, azimuth_factor, polar_factor, cartesian_coordinates);

        if (!isWithinShell(cartesian_coordinates)) {
            LOG_ERROR("Transformed point is outside the spherical shell: (x, y, z) = {}", cartesian_coordinates);
            throw std::runtime_error("Transformed point is outside the spherical shell)");
        }

        return {cartesian_coordinates[0], cartesian_coordinates[1], cartesian_coordinates[2]};
    }

    Eigen::Vector3d SphericalShellTransformer::sphericalToCartesian(const double radius, const double azimuth,
                                                                    const double polar) noexcept {
        return {
                radius * std::sin(polar) * std::cos(azimuth),
                radius * std::sin(polar) * std::sin(azimuth),
                radius * std::cos(polar)
        };
    }

    bool SphericalShellTransformer::isWithinShell(const Eigen::Vector3d &point) const noexcept {
        const double distance = point.norm();
        return (distance >= inner_radius_ && distance <= outer_radius_);
    }

    void SphericalShellTransformer::validate(const std::vector<double> &sample) const {
        try {
            Transformer::validate(sample);
        } catch (const std::invalid_argument &e) {
            LOG_ERROR("Invalid sample: {}", sample);
            throw std::invalid_argument(fmt::format("Invalid sample: {}", e.what()));
        }
        LOG_TRACE("Validated sample: (radius_factor, azimuth_factor, polar_factor) = {}", sample);
    }

}
