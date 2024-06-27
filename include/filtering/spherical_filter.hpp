// File: filtering/spherical_filter.hpp

#ifndef SPHERICAL_FILTER_HPP
#define SPHERICAL_FILTER_HPP

#include "filtering/filter.hpp"

#include <Eigen/Core>
#include <vector>

namespace filtering {

    class SphericalFilter : public Filter {
    public:
        SphericalFilter(double radius, double thickness_ratio);

        [[nodiscard]] std::vector<std::vector<double> > filter(const std::vector<std::vector<double> > &points,
                                                               double threshold) const override;

        [[nodiscard]] std::string getName() const override;

    private:
        double radius_;
        double inner_radius_;
        double outer_radius_;

        void calculateShellBounds(double thickness_ratio);

        [[nodiscard]] bool isWithinShell(const Eigen::Vector3f &point) const;
    };

}

#endif //SPHERICAL_FILTER_HPP
