// File: sampling/spherical_transformer.hpp

#ifndef SPHERICAL_TRANSFORMER_HPP
#define SPHERICAL_TRANSFORMER_HPP

#include <cmath>
#include <Eigen/Dense>

#include "sampling/transformer.hpp"
#include "common/logging/logger.hpp"

namespace sampling {

    class SphericalShellTransformer final : public Transformer<double> {
    public:
        SphericalShellTransformer(double inner_radius, double outer_radius);

        [[nodiscard]] std::vector<double> transform(const std::vector<double> &sample) const override;

        void validate(const std::vector<double> &sample) const override;

        constexpr static double PI() noexcept { return 3.141592653589793; }

    private:
        const double inner_radius_;
        const double outer_radius_;

        static Eigen::Vector3d sphericalToCartesian(double radius, double azimuth, double polar) noexcept;

        [[nodiscard]] bool isWithinShell(const Eigen::Vector3d &point) const noexcept;

    };

}

#endif //SPHERICAL_TRANSFORMER_HPP
