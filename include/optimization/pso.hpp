// File: optimization/particle_swarm_optimization.hpp

#ifndef PARTICLE_SWARM_OPTIMIZATION_HPP
#define PARTICLE_SWARM_OPTIMIZATION_HPP

#include <Eigen/Core>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "common/logging/logger.hpp"
#include "core/camera.hpp"
#include "core/perception.hpp"
#include "optimization/levenberg_marquardt.hpp"
#include "types/image.hpp"
#include "types/viewpoint.hpp"

class ParticleSwarmOptimization {
public:
    struct Particle {
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Vector3d best_position;
        double best_score{};
    };

    struct Parameters {
        int swarm_size, max_iterations, local_search_iterations;
        double inertia_weight, cognitive_coefficient, social_coefficient;
        double radius, tolerance; // percentage [0.0 - 1.0]
        double inertia_min, inertia_max; // Adaptive inertia weight range
        int early_termination_window; // Window for checking early termination
        double early_termination_threshold; // Improvement threshold for early termination
        double velocity_max; // Maximum velocity for normalization
    };

    explicit ParticleSwarmOptimization(const Parameters &params, const Image<> &target_image,
                                       const std::shared_ptr<processing::image::ImageComparator> &comparator) :
        params_(params), target_image_(target_image), comparator_(comparator),
        lm_optimizer_(optimization::LevenbergMarquardt<double, 3>()) {
        initializeSwarm();
    }

    ViewPoint<> optimize() {
        std::vector recent_best_scores(params_.early_termination_window, global_best_score_);
        for (int iteration = 0; iteration < params_.max_iterations; ++iteration) {
            const double inertia_weight = params_.inertia_min + (params_.inertia_max - params_.inertia_min) *
                                                                        (params_.max_iterations - iteration) /
                                                                        params_.max_iterations;

            for (auto &particle: swarm_) {
                updateParticle(particle, inertia_weight);
                const double score = evaluateParticle(particle.position);

                LOG_DEBUG("Iteration {}: Point = {}, Score = {}", iteration, particle.position.transpose(), score);

                if (score > particle.best_score) {
                    particle.best_score = score;
                    particle.best_position = particle.position;
                }

                if (score > global_best_score_) {
                    global_best_score_ = score;
                    global_best_position_ = particle.position;
                }
            }

            LOG_INFO("Iteration {}: Best score = {}", iteration, global_best_score_);

            recent_best_scores[iteration % params_.early_termination_window] = global_best_score_;
            if (iteration >= params_.early_termination_window) {
                const double improvement =
                        *std::ranges::max_element(recent_best_scores.begin(), recent_best_scores.end()) -
                        *std::ranges::min_element(recent_best_scores.begin(), recent_best_scores.end());
                if (improvement < params_.early_termination_threshold) {
                    LOG_INFO("Early termination at iteration {}: Improvement = {}", iteration, improvement);
                    break;
                }
            }

            if (isConverging()) {
                LOG_INFO("Switching to local optimization at iteration {}", iteration);
                localOptimization();
                break;
            }
        }
        return ViewPoint<>(global_best_position_, global_best_score_);
    }

private:
    Parameters params_;
    Image<> target_image_;
    std::shared_ptr<processing::image::ImageComparator> comparator_;
    std::vector<Particle> swarm_;
    Eigen::Vector3d global_best_position_;
    double global_best_score_ = -std::numeric_limits<double>::infinity();
    optimization::LevenbergMarquardt<double, 3> lm_optimizer_;

    void initializeSwarm() {
        swarm_.resize(params_.swarm_size);
        double step_size = 2 * params_.radius / std::cbrt(params_.swarm_size);
        int index = 0;

        for (double x = -params_.radius; x <= params_.radius; x += step_size) {
            for (double y = -params_.radius; y <= params_.radius; y += step_size) {
                for (double z = -params_.radius; z <= params_.radius; z += step_size) {
                    if (index >= params_.swarm_size)
                        break;
                    Eigen::Vector3d position(x, y, z);
                    if (isPositionValid(position)) {
                        swarm_[index].position = position;
                        swarm_[index].velocity = Eigen::Vector3d::Zero();
                        swarm_[index].best_position = position;
                        swarm_[index].best_score = evaluateParticle(position);
                        if (swarm_[index].best_score > global_best_score_) {
                            global_best_score_ = swarm_[index].best_score;
                            global_best_position_ = swarm_[index].best_position;
                        }
                        index++;
                    }
                }
            }
        }
    }

    void updateParticle(Particle &particle, const double inertia_weight) const {
        const Eigen::Vector3d inertia = inertia_weight * particle.velocity;
        const Eigen::Vector3d cognitive =
                params_.cognitive_coefficient *
                Eigen::Vector3d::Random().cwiseProduct(particle.best_position - particle.position);
        const Eigen::Vector3d social = params_.social_coefficient * Eigen::Vector3d::Random().cwiseProduct(
                                                                            global_best_position_ - particle.position);

        particle.velocity = inertia + cognitive + social;

        // Normalize velocity
        if (particle.velocity.norm() > params_.velocity_max) {
            particle.velocity = particle.velocity.normalized() * params_.velocity_max;
        }

        particle.position += particle.velocity;

        // Ensure the position norm constraint
        if (!isPositionValid(particle.position)) {
            particle.position = projectToValidPosition(particle.position);
        }
    }

    [[nodiscard]] bool isPositionValid(const Eigen::Vector3d &position) const {
        const double norm = position.norm();
        const double min_radius = params_.radius * (1.0 - params_.tolerance);
        const double max_radius = params_.radius * (1.0 + params_.tolerance);
        return norm >= min_radius && norm <= max_radius;
    }

    [[nodiscard]] Eigen::Vector3d projectToValidPosition(const Eigen::Vector3d &position) const {
        Eigen::Vector3d projected_position = position.normalized() * params_.radius;
        const double norm = projected_position.norm();
        const double min_radius = params_.radius * (1.0 - params_.tolerance);
        const double max_radius = params_.radius * (1.0 + params_.tolerance);

        if (norm < min_radius) {
            projected_position = position.normalized() * min_radius;
        } else if (norm > max_radius) {
            projected_position = position.normalized() * max_radius;
        }
        return projected_position;
    }

    double evaluateParticle(const Eigen::Vector3d &position) const {
        const auto viewpoint = ViewPoint<>::fromCartesian(position.x(), position.y(), position.z());
        const auto view = viewpoint.toView(Eigen::Vector3d::Zero());
        const Eigen::Matrix4d extrinsics = view.getPose();
        const cv::Mat rendered_view = core::Perception::render(extrinsics);

        const double score = comparator_->compare(target_image_.getImage(), rendered_view);
        return score;
    }

    [[nodiscard]] bool isConverging() const {
        double average_velocity = 0.0;
        for (const auto &particle: swarm_) {
            average_velocity += particle.velocity.norm();
        }
        average_velocity /= swarm_.size();
        return average_velocity < params_.early_termination_threshold;
    }

    void localOptimization() {
        // Start local optimization from the best global position found by PSO
        /*Eigen::Vector3d current_position = global_best_position_;

        optimization::LevenbergMarquardt<double, 3>::Options lm_options;
        lm_options.max_iterations = params_.local_search_iterations;
        auto result = lm_optimizer_.optimize(current_position, target_image_, comparator_);

        if (result.has_value()) {
            global_best_position_ = result->position;
            global_best_score_ = evaluateParticle(global_best_position_);
        }*/
    }
};

#endif // PARTICLE_SWARM_OPTIMIZATION_HPP
