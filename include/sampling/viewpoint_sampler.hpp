// File: sampling/viewpoint_sampler.hpp

#ifndef VIEWPOINT_SAMPLER_HPP
#define VIEWPOINT_SAMPLER_HPP

#include "spatial/octree.hpp"

template<FloatingPoint T = double>
class ViewpointSampler {
public:
    ViewpointSampler(const Eigen::Vector3<T> &center, T radius, T tolerance) :
        center_(center), radius_(radius), tolerance_(tolerance), rng_(std::random_device{}()) {}

    std::vector<ViewPoint<T>> samplePoints(const typename spatial::Octree<T>::Node &node) const {
        std::vector<ViewPoint<T>> points;
        std::uniform_real_distribution<T> dist(-0.5, 0.5);

        points.reserve(10);
        for (int i = 0; i < 10; ++i) {
            Eigen::Vector3<T> position =
                    node.center + node.size * Eigen::Vector3<T>(dist(rng_), dist(rng_), std::abs(dist(rng_)));
            position = projectToShell(position);
            points.emplace_back(position);
        }

        return points;
    }

    void addPointsAroundBest(typename spatial::Octree<T>::Node &node, const ViewPoint<T> &best_viewpoint) {
        constexpr int extra_points = 6;
        const T radius = node.size * 0.1;
        std::normal_distribution<T> dist(0, radius);
        for (int i = 0; i < extra_points; ++i) {
            Eigen::Vector3<T> offset(dist(rng_), dist(rng_), dist(rng_));
            node.points.emplace_back(best_viewpoint.getPosition() + offset);
        }
    }

    Eigen::Vector3<T> projectToShell(const Eigen::Vector3<T> &point,
                                     const ViewPoint<T> *best_viewpoint = nullptr) const {
        if (!radius_ || !tolerance_) {
            return point;
        }

        Eigen::Vector3<T> direction = point - center_;
        T distance = direction.norm();
        T min_radius = radius_ * (1 - tolerance_);
        T max_radius = radius_ * (1 + tolerance_);

        // Convert best_viewpoint to spherical coordinates if provided
        Eigen::Vector3<T> best_direction;
        T best_r = 0, best_theta = 0, best_phi = 0;
        if (best_viewpoint) {
            best_direction = best_viewpoint->getPosition() - center_;
            best_r = best_direction.norm();
            best_theta = std::atan2(best_direction.y(), best_direction.x());
            best_phi = std::acos(best_direction.z() / best_r);
        }

        // Convert point to spherical coordinates
        T r = distance;
        T theta = std::atan2(direction.y(), direction.x());
        T phi = std::acos(direction.z() / r);

        // Restrict to upper hemisphere
        phi = std::min(phi, static_cast<T>(M_PI_2));

        // Calculate the maximum allowed angular distance based on the current best score
        bool restrict_vicinity = config::get("octree.restrict_vicinity", false);
        T max_angular_distance = M_PI_2;

        if (restrict_vicinity && best_viewpoint) {
            T vicinity_multiplier = config::get("octree.vicinity_multiplier", 0.5);
            if (vicinity_multiplier <= 0 || vicinity_multiplier > 1) {
                LOG_WARN("Invalid vicinity multiplier: {}. Disabling vicinity restriction.", vicinity_multiplier);
            } else {
                max_angular_distance = M_PI_2 * (1 - best_viewpoint->getScore()) * vicinity_multiplier;
            }
        }

        max_angular_distance = std::max(max_angular_distance,
                                        config::get("octree.min_vicinity", T(M_PI_2 / 6))); // Minimum search area

        // Restrict theta and phi to be within max_angular_distance of the best viewpoint
        if (best_viewpoint) {
            T delta_theta = std::abs(theta - best_theta);
            if (delta_theta > M_PI) {
                delta_theta = 2 * M_PI - delta_theta;
            }
            if (delta_theta > max_angular_distance) {
                theta = best_theta + max_angular_distance * (theta > best_theta ? 1 : -1);
            }

            T delta_phi = std::abs(phi - best_phi);
            if (delta_phi > max_angular_distance) {
                phi = best_phi + max_angular_distance * (phi > best_phi ? 1 : -1);
            }
        }

        // Convert back to Cartesian coordinates
        Eigen::Vector3<T> projected_point(r * std::sin(phi) * std::cos(theta), r * std::sin(phi) * std::sin(theta),
                                          r * std::cos(phi));

        // Ensure the radius is within the allowed range
        if (r < min_radius) {
            projected_point *= min_radius / r;
        } else if (r > max_radius) {
            projected_point *= max_radius / r;
        }

        return center_ + projected_point;
    }


private:
    Eigen::Vector3<T> center_;
    T radius_, tolerance_;
    mutable std::mt19937 rng_;
};

#endif // VIEWPOINT_SAMPLER_HPP
