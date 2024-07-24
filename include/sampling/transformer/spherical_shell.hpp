// File: sampling/spherical_shell.hpp

#ifndef SPHERICAL_TRANSFORMER_HPP
#define SPHERICAL_TRANSFORMER_HPP

#include <Eigen/Dense>
#include <cmath>
#include "sampling/transformer.hpp"

template<typename T = double>
class SphericalShellTransformer final : public Transformer<T> {
public:
    SphericalShellTransformer(T radius, T tolerance) : radius_(radius), tolerance_(tolerance), epsilon_(1e-6) {
        if (radius <= 0 || tolerance <= 0) {
            LOG_ERROR("Invalid radius or tolerance values. Received: {} and {}.", radius, tolerance);
            throw std::invalid_argument("Invalid radius or tolerance values.");
        }
        inner_radius_ = radius_ * (1.0 - tolerance_);
        outer_radius_ = radius_ * (1.0 + tolerance_);

        LOG_DEBUG("Spherical shell transformer initialized with radius {} and tolerance {}.", radius, tolerance);
        LOG_DEBUG("Calculated inner radius: {}, outer radius: {}.", inner_radius_, outer_radius_);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> transform(const Eigen::Matrix<T, Eigen::Dynamic, 1> &sample) const override {
        this->validate(sample);

        LOG_DEBUG("Transforming sample ({}, {}, {}).", sample(0), sample(1), sample(2));

        T radius_factor = sample(0);
        T azimuth_factor = sample(1);
        T polar_factor = sample(2);

        T adjusted_radius = inner_radius_ + (outer_radius_ - inner_radius_) * radius_factor;
        T azimuth = 2.0 * M_PI * azimuth_factor;
        // T polar = std::acos(1.0 - 2.0 * polar_factor); // Ensure valid input to acos
        T polar = std::acos(1.0 - 2.0 * std::clamp(polar_factor, T(0.0), T(1.0))); // Ensure valid input to acos

        LOG_DEBUG("Spherical coordinates: r = {}, theta = {}, phi = {}.", adjusted_radius, azimuth, polar);

        Eigen::Matrix<T, Eigen::Dynamic, 1> transformed = sphericalToCartesian(adjusted_radius, azimuth, polar);

        if (!isWithinShell(transformed)) {
            LOG_ERROR("Transformed point is outside the spherical shell. Transformed: ({}, {}, {}).", transformed(0),
                      transformed(1), transformed(2));
            throw std::runtime_error("Transformed point is outside the spherical shell.");
        }

        LOG_DEBUG("Sample transformed to ({}, {}, {}).", transformed(0), transformed(1), transformed(2));

        return transformed;
    }

protected:
    void validate(const Eigen::Matrix<T, Eigen::Dynamic, 1> &sample) const override {
        Transformer<T>::validate(sample);
        if (sample.size() != 3) {
            LOG_ERROR("Sample must have exactly 3 dimensions for spherical transformation. Received: {}.",
                      sample.size());
            throw std::invalid_argument("Sample must have exactly three dimensions for spherical transformation.");
        }

        for (int i = 0; i < sample.size(); ++i) {
            if (sample(i) < 0.0 || sample(i) > 1.0) {
                LOG_ERROR("Sample values must be within [0, 1]. Received: {}.", sample(i));
                throw std::invalid_argument("Sample values must be within [0, 1].");
            }
        }
        LOG_DEBUG("Sample ({}, {}, {}) validated successfully.", sample(0), sample(1), sample(2));
    }

private:
    T radius_;
    T tolerance_;
    T inner_radius_;
    T outer_radius_;
    T epsilon_;

    Eigen::Matrix<T, Eigen::Dynamic, 1> sphericalToCartesian(T radius, T azimuth, T polar) const noexcept {
        Eigen::Matrix<T, Eigen::Dynamic, 1> cartesian(3);
        cartesian << radius * std::sin(polar) * std::cos(azimuth), radius * std::sin(polar) * std::sin(azimuth),
                radius * std::cos(polar);
        return cartesian;
    }


    bool isWithinShell(const Eigen::Matrix<T, Eigen::Dynamic, 1> &point) const noexcept {
        LOG_DEBUG("Checking if point ({}, {}, {}) is within the spherical shell.", point(0), point(1), point(2));
        T norm = point.norm();
        LOG_DEBUG("Point norm: {}.", norm);
        return norm >= (inner_radius_ - epsilon_) && norm <= (outer_radius_ + epsilon_);
    }
};

#endif // SPHERICAL_TRANSFORMER_HPP
