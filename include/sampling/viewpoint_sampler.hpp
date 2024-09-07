// File: sampling/viewpoint_sampler.hpp

#ifndef VIEWPOINT_SAMPLER_HPP
#define VIEWPOINT_SAMPLER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/logging/logger.hpp"
#include "optimization/acquisition.hpp"
#include "optimization/gaussian/gpr.hpp"
#include "spatial/octree.hpp"

template<FloatingPoint T = double>
class ViewpointSampler {
public:
    ViewpointSampler(const Eigen::Vector3<T> &center, T radius, T tolerance,
                     const optimization::GPR<optimization::kernel::Matern52<T>> &gpr,
                     optimization::Acquisition<T> &acquisition) :
        center_(center), radius_(radius), tolerance_(tolerance), gpr_(gpr), acquisition_(acquisition),
        rng_(std::random_device{}()), uniform_dist_(0.0, 1.0), normal_dist_(0.0, 1.0) {
        try {
            min_radius_ = radius_ * (1 - tolerance_);
            max_radius_ = radius_ * (1 + tolerance_);
            if (min_radius_ <= 0 || max_radius_ <= min_radius_) {
                throw std::invalid_argument("Invalid radius or tolerance values");
            }
        } catch (const std::exception &e) {
            LOG_ERROR("Error initializing ViewpointSampler: {}", e.what());
            min_radius_ = std::max(T(0.1), radius_ * 0.5);
            max_radius_ = radius_ * 1.5;
            LOG_WARN("Using fallback values: min_radius = {}, max_radius = {}", min_radius_, max_radius_);
        }
    }

    std::vector<ViewPoint<T>> samplePoints(const typename spatial::Octree<T>::Node &node,
                                           const ViewPoint<T> *best_viewpoint = nullptr) {
        try {
            std::vector<ViewPoint<T>> candidates = generateCandidates(node, best_viewpoint);
            std::vector<ViewPoint<T>> selected_points = selectBestCandidates(candidates);

            LOG_INFO("Generated {} candidates, selected {} best points", candidates.size(), selected_points.size());
            return selected_points;
        } catch (const std::exception &e) {
            LOG_ERROR("Error in samplePoints: {}", e.what());
            return fallbackSampling(node);
        }
    }

    void addPointsAroundBest(typename spatial::Octree<T>::Node &node, const ViewPoint<T> &best_viewpoint) {
        try {
            std::vector<ViewPoint<T>> candidates = generateCandidatesAroundBest(node, best_viewpoint);
            std::vector<ViewPoint<T>> selected_points = selectBestCandidates(candidates);

            for (const auto &point: selected_points) {
                if (isDiverseEnough(node.points, point)) {
                    node.points.push_back(point);
                }
            }

            LOG_INFO("Added {} diverse points around best viewpoint", selected_points.size());
        } catch (const std::exception &e) {
            LOG_ERROR("Error in addPointsAroundBest: {}", e.what());
            fallbackAddPointsAroundBest(node, best_viewpoint);
        }
    }

private:
    Eigen::Vector3<T> center_;
    T radius_, tolerance_, min_radius_, max_radius_;
    const optimization::GPR<optimization::kernel::Matern52<T>> &gpr_;
    optimization::Acquisition<T> &acquisition_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<T> uniform_dist_;
    mutable std::normal_distribution<T> normal_dist_;

    std::vector<ViewPoint<T>> generateCandidates(const typename spatial::Octree<T>::Node &node,
                                                 const ViewPoint<T> *best_viewpoint) const {
        constexpr int num_candidates = 50; // Increased for better coverage
        std::vector<ViewPoint<T>> candidates;
        candidates.reserve(num_candidates);

        for (int i = 0; i < num_candidates; ++i) {
            Eigen::Vector3<T> position = generateCandidatePoint(node, best_viewpoint);
            candidates.emplace_back(position);
        }

        return candidates;
    }

    std::vector<ViewPoint<T>> generateCandidatesAroundBest(const typename spatial::Octree<T>::Node &node,
                                                           const ViewPoint<T> &best_viewpoint) const {
        constexpr int num_candidates = 30;
        std::vector<ViewPoint<T>> candidates;
        candidates.reserve(num_candidates);

        for (int i = 0; i < num_candidates; ++i) {
            Eigen::Vector3<T> offset = sampleOffsetAroundBest(node.size * 0.15);
            Eigen::Vector3<T> new_point = projectToShell(best_viewpoint.getPosition() + offset, &best_viewpoint);
            candidates.emplace_back(new_point);
        }

        return candidates;
    }

    std::vector<ViewPoint<T>> selectBestCandidates(const std::vector<ViewPoint<T>> &candidates) const {
        std::vector<std::pair<T, ViewPoint<T>>> ranked_candidates;
        ranked_candidates.reserve(candidates.size());

        T best_score = std::numeric_limits<T>::lowest();
        for (const auto &candidate: candidates) {
            auto [mean, std_dev] = gpr_.predict(candidate.getPosition());
            T acquisition_value = acquisition_.compute(candidate.getPosition(), mean, std_dev);
            ranked_candidates.emplace_back(acquisition_value, candidate);
            best_score = std::max(best_score, mean);
        }

        // Sort candidates by acquisition value
        std::sort(ranked_candidates.begin(), ranked_candidates.end(), std::greater<std::pair<T, ViewPoint<T>>>());

        // Select top 10 candidates, ensuring diversity and improvement
        std::vector<ViewPoint<T>> selected_points;
        selected_points.reserve(10);

        for (const auto &[value, point]: ranked_candidates) {
            if (isDiverseEnough(selected_points, point) && isLikelyImprovement(point, best_score)) {
                selected_points.push_back(point);
                if (selected_points.size() == 10)
                    break;
            }
        }

        return selected_points;
    }

    Eigen::Vector3<T> generateCandidatePoint(const typename spatial::Octree<T>::Node &node,
                                             const ViewPoint<T> *best_viewpoint) const {
        int attempts = 0;
        constexpr int max_attempts = 10;

        while (attempts < max_attempts) {
            Eigen::Vector3<T> raw_point;
            if (best_viewpoint && uniform_dist_(rng_) < 0.7) {
                raw_point = sampleNearBestViewpoint(*best_viewpoint, node.size);
            } else {
                raw_point = sampleAdaptiveInNode(node);
            }

            Eigen::Vector3<T> projected_point = projectToShell(raw_point, best_viewpoint);

            if (isValidPoint(projected_point)) {
                return projected_point;
            }

            attempts++;
        }

        LOG_WARN("Failed to generate valid point after {} attempts. Using fallback method.", max_attempts);
        return fallbackProjection(node.center);
    }


    Eigen::Vector3<T> sampleNearBestViewpoint(const ViewPoint<T> &best_viewpoint, T node_size) const {
        std::normal_distribution<T> dist(0, node_size * 0.1);
        Eigen::Vector3<T> offset(dist(rng_), dist(rng_), std::abs(dist(rng_)));
        return best_viewpoint.getPosition() + offset;
    }

    Eigen::Vector3<T> sampleAdaptiveInNode(const typename spatial::Octree<T>::Node &node) const {
        T adaptive_size = node.size * (1 - std::min(node.max_acquisition, T(0.9)));
        return node.center + adaptive_size * Eigen::Vector3<T>(uniform_dist_(rng_) - 0.5, uniform_dist_(rng_) - 0.5,
                                                               std::abs(uniform_dist_(rng_) - 0.5));
    }

    Eigen::Vector3<T> sampleOffsetAroundBest(T radius) const {
        Eigen::Vector3<T> offset;
        do {
            offset = Eigen::Vector3<T>(std::normal_distribution<T>(0, radius)(rng_),
                                       std::normal_distribution<T>(0, radius)(rng_),
                                       std::abs(std::normal_distribution<T>(0, radius)(rng_)));
        } while (offset.norm() > radius);

        return offset;
    }

    /*Eigen::Vector3<T> projectToShell(const Eigen::Vector3<T> &point, const ViewPoint<T> *best_viewpoint) const {
        try {
            Eigen::Vector3<T> direction = point - center_;
            T distance = direction.norm();

            if (distance < std::numeric_limits<T>::epsilon()) {
                LOG_WARN("Point too close to center. Using fallback method.");
                return fallbackProjection(point);
            }


            // Convert to spherical coordinates
            T r = std::clamp(distance, min_radius_, max_radius_);
            T theta = std::atan2(direction.y(), direction.x());
            T phi = std::acos(std::clamp(direction.z() / distance, T(-1), T(1)));

            // Restrict to upper hemisphere
            phi = std::clamp(phi, T(0), T(M_PI_2));

            // Apply adaptive constraints based on GPR predictions
            if (best_viewpoint) {
                Eigen::Vector3<T> best_direction = best_viewpoint->getPosition() - center_;
                T best_r = best_direction.norm();
                T best_theta = std::atan2(best_direction.y(), best_direction.x());
                T best_phi = std::acos(std::clamp(best_direction.z() / best_r, T(-1), T(1)));

                auto [mean, std_dev] = gpr_.predict(point);
                T prediction_confidence = 1 / (1 + std_dev);

                // Adaptive biasing towards best_viewpoint
                T bias_strength = 0.3 * prediction_confidence;
                r = r * (1 - bias_strength) + best_r * bias_strength;
                theta = theta * (1 - bias_strength) + best_theta * bias_strength;
                phi = phi * (1 - bias_strength) + best_phi * bias_strength;
            }

            // Add controlled randomness to avoid exact boundary placement
            T r_noise = (uniform_dist_(rng_) - 0.5) * (max_radius_ - min_radius_) * 0.05;
            T theta_noise = (uniform_dist_(rng_) - 0.5) * M_PI * 0.05;
            T phi_noise = (uniform_dist_(rng_) - 0.5) * M_PI_2 * 0.05;

            r = std::clamp(r + r_noise, min_radius_, max_radius_);
            theta += theta_noise;
            phi = std::clamp(phi + phi_noise, T(0), T(M_PI_2));

            // Convert back to Cartesian coordinates
            Eigen::Vector3<T> result =
                    center_ + Eigen::Vector3<T>(r * std::sin(phi) * std::cos(theta),
                                                r * std::sin(phi) * std::sin(theta), r * std::cos(phi));

            if (!isValidPoint(result)) {
                LOG_WARN("Projected point is invalid. Using fallback method.");
                return fallbackProjection(point);
            }
            return result;

        } catch (const std::exception &e) {
            LOG_ERROR("Error in projectToShell: {}", e.what());
            return fallbackProjection(point);
        }
    }*/

    Eigen::Vector3<T> projectToShell(const Eigen::Vector3<T> &point, const ViewPoint<T> *best_viewpoint) const {
        Eigen::Vector3<T> direction = point - center_;
        T distance = direction.norm();

        if (distance < std::numeric_limits<T>::epsilon()) {
            return fallbackProjection(point);
        }

        // Convert to spherical coordinates
        T r = std::clamp(distance, min_radius_, max_radius_);
        T theta = std::atan2(direction.y(), direction.x());
        T phi = std::acos(std::clamp(direction.z() / distance, T(-1), T(1)));

        // Restrict to upper hemisphere
        phi = std::clamp(phi, T(0), T(M_PI_2));

        // Apply adaptive constraints based on GPR predictions
        if (best_viewpoint) {
            Eigen::Vector3<T> best_direction = best_viewpoint->getPosition() - center_;
            T best_r = best_direction.norm();
            T best_theta = std::atan2(best_direction.y(), best_direction.x());
            T best_phi = std::acos(std::clamp(best_direction.z() / best_r, T(-1), T(1)));

            auto [mean, std_dev] = gpr_.predict(point);
            if (std::isnan(mean) || std::isnan(std_dev)) {
                LOG_WARN("GPR prediction returned NaN. Using original point.");
            } else {
                T prediction_confidence = 1 / (1 + std_dev);
                T bias_strength = 0.3 * prediction_confidence;
                r = r * (1 - bias_strength) + best_r * bias_strength;
                theta = theta * (1 - bias_strength) + best_theta * bias_strength;
                phi = phi * (1 - bias_strength) + best_phi * bias_strength;
            }
        }

        // Add controlled randomness to avoid exact boundary placement
        T r_noise = (uniform_dist_(rng_) - 0.5) * (max_radius_ - min_radius_) * 0.05;
        T theta_noise = (uniform_dist_(rng_) - 0.5) * M_PI * 0.05;
        T phi_noise = (uniform_dist_(rng_) - 0.5) * M_PI_2 * 0.05;

        r = std::clamp(r + r_noise, min_radius_, max_radius_);
        theta += theta_noise;
        phi = std::clamp(phi + phi_noise, T(0), T(M_PI_2));

        // Convert back to Cartesian coordinates
        Eigen::Vector3<T> result = center_ + Eigen::Vector3<T>(r * std::sin(phi) * std::cos(theta),
                                                               r * std::sin(phi) * std::sin(theta), r * std::cos(phi));

        return result;
    }

    bool isDiverseEnough(const std::vector<ViewPoint<T>> &existing_points, const ViewPoint<T> &new_point) const {
        const T diversity_threshold = radius_ * 0.05; // 5% of the radius for tighter packing
        for (const auto &point: existing_points) {
            if ((point.getPosition() - new_point.getPosition()).norm() < diversity_threshold) {
                return false;
            }
        }
        return true;
    }

    bool isLikelyImprovement(const ViewPoint<T> &point, T best_score) const {
        auto [mean, std_dev] = gpr_.predict(point.getPosition());
        T z_score = (best_score - mean) / std_dev;
        T improvement_probability = 1 - 0.5 * (1 + std::erf(z_score / std::sqrt(2)));
        return improvement_probability > 0.3; // 30% chance of improvement
    }

    /*bool isValidPoint(const Eigen::Vector3<T> &point) const {
        return point.allFinite() && (point - center_).norm() >= min_radius_ && (point - center_).norm() <= max_radius_;
    }*/

    bool isValidPoint(const Eigen::Vector3<T> &point) const {
        if (!point.allFinite()) {
            return false;
        }

        Eigen::Vector3<T> direction = point - center_;
        T distance = direction.norm();

        if (distance < min_radius_ || distance > max_radius_) {
            return false;
        }

        // Ensure the point is in the upper hemisphere
        if (direction.z() < 0) {
            return false;
        }

        return true;
    }

    std::vector<ViewPoint<T>> fallbackSampling(const typename spatial::Octree<T>::Node &node) const {
        LOG_WARN("Using fallback sampling method");
        std::vector<ViewPoint<T>> fallback_points;
        fallback_points.reserve(10);
        for (int i = 0; i < 10; ++i) {
            Eigen::Vector3<T> position =
                    node.center + node.size * Eigen::Vector3<T>(uniform_dist_(rng_) - 0.5, uniform_dist_(rng_) - 0.5,
                                                                std::abs(uniform_dist_(rng_) - 0.5));
            fallback_points.emplace_back(projectToShell(position, nullptr));
        }
        return fallback_points;
    }

    void fallbackAddPointsAroundBest(typename spatial::Octree<T>::Node &node, const ViewPoint<T> &best_viewpoint) {
        LOG_WARN("Using fallback method to add points around best viewpoint");
        constexpr int extra_points = 5;
        const T radius = node.size * 0.1;
        std::normal_distribution<T> dist(0, radius);
        for (int i = 0; i < extra_points; ++i) {
            Eigen::Vector3<T> offset(dist(rng_), dist(rng_), std::abs(dist(rng_)));
            Eigen::Vector3<T> new_point = projectToShell(best_viewpoint.getPosition() + offset, &best_viewpoint);
            node.points.emplace_back(new_point);
        }
    }

    /*Eigen::Vector3<T> fallbackProjection(const Eigen::Vector3<T> &point) const {
        LOG_WARN("Using fallback projection method");
        Eigen::Vector3<T> direction = point - center_;
        if (direction.norm() < std::numeric_limits<T>::epsilon()) {
            direction = Eigen::Vector3<T>(uniform_dist_(rng_), uniform_dist_(rng_), std::abs(uniform_dist_(rng_)));
        }
        direction.normalize();
        T r = min_radius_ + uniform_dist_(rng_) * (max_radius_ - min_radius_);
        return center_ + r * direction;
    }*/

    Eigen::Vector3<T> fallbackProjection(const Eigen::Vector3<T> &point) const {
        LOG_WARN("Using fallback projection method");
        Eigen::Vector3<T> direction = point - center_;
        if (direction.norm() < std::numeric_limits<T>::epsilon()) {
            direction = Eigen::Vector3<T>(uniform_dist_(rng_), uniform_dist_(rng_), std::abs(uniform_dist_(rng_)));
        }
        direction.normalize();
        T r = min_radius_ + uniform_dist_(rng_) * (max_radius_ - min_radius_);
        Eigen::Vector3<T> result = center_ + r * direction;

        // Ensure the point is in the upper hemisphere
        result.z() = std::abs(result.z());

        return result;
    }
};

#endif // VIEWPOINT_SAMPLER_HPP
