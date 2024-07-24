// File: optimization/particle_swarm_optimization.hpp

#ifndef PARTICLE_SWARM_OPTIMIZATION_HPP
#define PARTICLE_SWARM_OPTIMIZATION_HPP

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "common/logging/logger.hpp"
#include "core/perception.hpp"
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
        int swarm_size, max_iterations;
        double inertia_weight, cognitive_coefficient, social_coefficient;
        double radius, tolerance; // percentage [0.0 - 1.0]
    };

    explicit ParticleSwarmOptimization(const Parameters &params, const Image<> &target_image,
                                       std::unique_ptr<processing::image::ImageComparator> comparator) :
        params_(params), target_image_(target_image), comparator_(std::move(comparator)) {
        initializeSwarm();
    }

    ViewPoint<> optimize() {
        for (int iteration = 0; iteration < params_.max_iterations; ++iteration) {
            for (auto &particle: swarm_) {
                updateParticle(particle);
                double score = evaluateParticle(particle);

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
        }
        return ViewPoint<>(global_best_position_, global_best_score_);
    }

private:
    Parameters params_;
    Image<> target_image_;
    std::unique_ptr<processing::image::ImageComparator> comparator_;
    std::vector<Particle> swarm_;
    Eigen::Vector3d global_best_position_;
    double global_best_score_ = -std::numeric_limits<double>::infinity();

    void initializeSwarm() {
        swarm_.resize(params_.swarm_size);
        for (auto &particle: swarm_) {
            do {
                particle.position = Eigen::Vector3d::Random();
            } while (!isPositionValid(particle.position));
            particle.velocity = Eigen::Vector3d::Random() * 0.1;
            particle.best_position = particle.position;
            particle.best_score = evaluateParticle(particle);
            if (particle.best_score > global_best_score_) {
                global_best_score_ = particle.best_score;
                global_best_position_ = particle.best_position;
            }
        }
    }

    void updateParticle(Particle &particle) const {
        Eigen::Vector3d inertia = params_.inertia_weight * particle.velocity;
        Eigen::Vector3d cognitive = params_.cognitive_coefficient *
                                    Eigen::Vector3d::Random().cwiseProduct(particle.best_position - particle.position);
        Eigen::Vector3d social = params_.social_coefficient *
                                 Eigen::Vector3d::Random().cwiseProduct(global_best_position_ - particle.position);

        particle.velocity = inertia + cognitive + social;
        particle.position += particle.velocity;

        // Ensure the position norm constraint
        if (!isPositionValid(particle.position)) {
            particle.position = projectToValidPosition(particle.position);
        }
    }

    [[nodiscard]] bool isPositionValid(const Eigen::Vector3d &position) const {
        double norm = position.norm();
        double min_radius = params_.radius * (1.0 - params_.tolerance);
        double max_radius = params_.radius * (1.0 + params_.tolerance);
        return norm >= min_radius && norm <= max_radius;
    }

    [[nodiscard]] Eigen::Vector3d projectToValidPosition(const Eigen::Vector3d &position) const {
        Eigen::Vector3d projected_position = position.normalized() * params_.radius;
        double norm = projected_position.norm();
        double min_radius = params_.radius * (1.0 - params_.tolerance);
        double max_radius = params_.radius * (1.0 + params_.tolerance);

        if (norm < min_radius) {
            projected_position = position.normalized() * min_radius;
        } else if (norm > max_radius) {
            projected_position = position.normalized() * max_radius;
        }
        return projected_position;
    }

    double evaluateParticle(Particle &particle) {
        auto viewpoint =
                ViewPoint<>::fromCartesian(particle.position.x(), particle.position.y(), particle.position.z());
        auto view = viewpoint.toView(Eigen::Vector3d::Zero());
        Eigen::Matrix4d extrinsics = view.getPose();
        cv::Mat rendered_view = core::Perception::render(extrinsics);

        double score = comparator_->compare(target_image_.getImage(), rendered_view);
        return score;
    }
};

#endif // PARTICLE_SWARM_OPTIMIZATION_HPP
