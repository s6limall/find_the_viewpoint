#ifndef ADVANCED_PARTICLE_SWARM_OPTIMIZATION_HPP
#define ADVANCED_PARTICLE_SWARM_OPTIMIZATION_HPP

#include <Eigen/Core>
#include <algorithm>
#include <execution>
#include <memory>
#include <random>
#include <vector>
#include "common/logging/logger.hpp"
#include "optimization/levenberg_marquardt.hpp"

namespace optimization {

    template<typename Scalar = double, int Dim = 3>
    class AdvancedParticleSwarmOptimization {
    public:
        using VectorType = Eigen::Matrix<Scalar, Dim, 1>;
        using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Dim>;
        using ErrorFunction = std::function<Scalar(const VectorType &)>;
        using JacobianFunction = std::function<JacobianType(const VectorType &)>;
        using ConstraintFunction = std::function<Scalar(const VectorType &)>;

        struct Particle {
            VectorType position;
            VectorType velocity;
            VectorType best_position;
            Scalar best_score;
            std::vector<int> neighbors;
        };

        struct Parameters {
            int swarm_size = 100;
            int max_iterations = 1000;
            int local_search_iterations = 50;
            Scalar inertia_min = 0.4;
            Scalar inertia_max = 0.9;
            Scalar cognitive_coefficient = 2.0;
            Scalar social_coefficient = 2.0;
            Scalar velocity_max = 0.1;
            Scalar diversity_threshold = 0.01;
            int stagnation_threshold = 20;
            Scalar penalty_coefficient = 1e3;
            int neighborhood_size = 5;
            Scalar search_radius;
            Scalar perturbation_amount = 0.05;
            bool use_obl = true;
            bool use_dynamic_topology = true;
            bool use_diversity_guided = true;
        };

        AdvancedParticleSwarmOptimization(const Parameters &params, const ErrorFunction &error_func,
                                          const JacobianFunction &jacobian_func,
                                          const std::vector<ConstraintFunction> &constraints,
                                          const VectorType &lower_bound, const VectorType &upper_bound) :
            params_(params), error_func_(error_func), jacobian_func_(jacobian_func), constraints_(constraints),
            lower_bound_(lower_bound), upper_bound_(upper_bound), lm_optimizer_(LevenbergMarquardt<Scalar, Dim>()),
            rng_(std::random_device{}()) {
            initializeSwarm();
        }

        VectorType optimize() {
            int stagnation_counter = 0;
            Scalar prev_best_score = std::numeric_limits<Scalar>::lowest();

            for (int iteration = 0; iteration < params_.max_iterations; ++iteration) {
                if (iteration % params_.stagnation_threshold == 0) {
                    regenerateSwarmAroundBest();
                }

                updateInertiaWeight(iteration);
                updateNeighborhoods();

                std::for_each(std::execution::par_unseq, swarm_.begin(), swarm_.end(),
                              [this](Particle &particle) { updateParticle(particle); });

                if (params_.use_obl)
                    applyOBL();
                if (params_.use_diversity_guided)
                    applyDiversityGuidedPSO();

                Scalar current_best_score = global_best_score_;
                if (std::abs(current_best_score - prev_best_score) < params_.diversity_threshold) {
                    stagnation_counter++;
                } else {
                    stagnation_counter = 0;
                }

                if (stagnation_counter >= params_.stagnation_threshold) {
                    LOG_INFO("Stagnation detected. Performing local search.");
                    localOptimization();
                    stagnation_counter = 0;
                }

                prev_best_score = current_best_score;
                LOG_INFO("Iteration {}: Best score = {}", iteration, global_best_score_);

                if (isConverged()) {
                    LOG_INFO("Convergence reached at iteration {}", iteration);
                    break;
                }
            }

            return global_best_position_;
        }

        Scalar evaluateParticle(const VectorType &position) const {
            Scalar score = -error_func_(position);
            Scalar penalty = 0.0;

            for (const auto &constraint: constraints_) {
                Scalar constraint_violation = std::max(Scalar(0), constraint(position));
                penalty += params_.penalty_coefficient * constraint_violation * constraint_violation;
            }

            return score - penalty;
        }

    private:
        Parameters params_;
        ErrorFunction error_func_;
        JacobianFunction jacobian_func_;
        std::vector<ConstraintFunction> constraints_;
        VectorType lower_bound_;
        VectorType upper_bound_;
        std::vector<Particle> swarm_;
        VectorType global_best_position_;
        Scalar global_best_score_ = std::numeric_limits<Scalar>::lowest();
        LevenbergMarquardt<Scalar, Dim> lm_optimizer_;
        std::mt19937 rng_;
        Scalar current_inertia_weight_;

        void initializeSwarm() {
            swarm_.resize(params_.swarm_size);
            Scalar radius = params_.search_radius;
            std::uniform_real_distribution<Scalar> dist(-radius, radius);

            for (auto &particle: swarm_) {
                VectorType random_position = VectorType::NullaryExpr(Dim, [&]() { return dist(rng_); });
                particle.position = random_position.normalized() * radius;
                particle.velocity = VectorType::Zero();
                particle.best_position = particle.position;
                particle.best_score = evaluateParticle(particle.position);

                if (particle.best_score > global_best_score_) {
                    global_best_score_ = particle.best_score;
                    global_best_position_ = particle.best_position;
                }
            }
        }


        void updateInertiaWeight(int iteration) {
            current_inertia_weight_ = params_.inertia_max -
                                      (params_.inertia_max - params_.inertia_min) * iteration / params_.max_iterations;
        }

        void updateNeighborhoods() {
            if (!params_.use_dynamic_topology)
                return;

            std::vector<int> indices(params_.swarm_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng_);

            for (int i = 0; i < params_.swarm_size; ++i) {
                swarm_[i].neighbors.clear();
                for (int j = 0; j < params_.neighborhood_size; ++j) {
                    swarm_[i].neighbors.push_back(indices[(i + j) % params_.swarm_size]);
                }
            }
        }

        void updateParticle(Particle &particle) {
            VectorType local_best_position = particle.best_position;
            Scalar local_best_score = particle.best_score;

            for (int neighbor: particle.neighbors) {
                if (swarm_[neighbor].best_score > local_best_score) {
                    local_best_score = swarm_[neighbor].best_score;
                    local_best_position = swarm_[neighbor].best_position;
                }
            }

            std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
            VectorType cognitive =
                    params_.cognitive_coefficient * dist(rng_) * (particle.best_position - particle.position);
            VectorType social = params_.social_coefficient * dist(rng_) * (local_best_position - particle.position);

            particle.velocity = current_inertia_weight_ * particle.velocity + cognitive + social;
            particle.velocity = particle.velocity.cwiseMin(params_.velocity_max).cwiseMax(-params_.velocity_max);

            particle.position += particle.velocity;
            particle.position = particle.position.cwiseMax(lower_bound_).cwiseMin(upper_bound_);

            Scalar score = evaluateParticle(particle.position);

            if (score > particle.best_score) {
                particle.best_score = score;
                particle.best_position = particle.position;

                if (score > global_best_score_) {
                    global_best_score_ = score;
                    global_best_position_ = particle.position;
                }
            }
        }


        void applyOBL() {
            for (auto &particle: swarm_) {
                VectorType opposite_position = lower_bound_ + upper_bound_ - particle.position;
                Scalar opposite_score = evaluateParticle(opposite_position);

                if (opposite_score > particle.best_score) {
                    particle.position = opposite_position;
                    particle.best_position = opposite_position;
                    particle.best_score = opposite_score;

                    if (opposite_score > global_best_score_) {
                        global_best_score_ = opposite_score;
                        global_best_position_ = opposite_position;
                    }
                }
            }
        }

        void applyDiversityGuidedPSO() {
            Scalar diversity = calculateSwarmDiversity();
            if (diversity < params_.diversity_threshold) {
                VectorType center_of_mass = VectorType::Zero();
                for (const auto &particle: swarm_) {
                    center_of_mass += particle.position;
                }
                center_of_mass /= params_.swarm_size;

                for (auto &particle: swarm_) {
                    particle.velocity +=
                            (particle.position - center_of_mass) * (1.0 - diversity / params_.diversity_threshold);
                }
            }
        }

        Scalar calculateSwarmDiversity() const {
            VectorType mean_position = VectorType::Zero();
            for (const auto &particle: swarm_) {
                mean_position += particle.position;
            }
            mean_position /= params_.swarm_size;

            Scalar diversity = 0.0;
            for (const auto &particle: swarm_) {
                diversity += (particle.position - mean_position).squaredNorm();
            }
            return std::sqrt(diversity / params_.swarm_size);
        }

        void regenerateSwarmAroundBest() {
            std::uniform_real_distribution<Scalar> dist(-params_.perturbation_amount, params_.perturbation_amount);
            for (auto &particle: swarm_) {
                VectorType random_direction = VectorType::NullaryExpr(Dim, [&]() { return dist(rng_); }).normalized();
                particle.position = global_best_position_ + random_direction * dist(rng_);
                particle.position = particle.position.cwiseMax(lower_bound_).cwiseMin(upper_bound_);
                particle.velocity = VectorType::Zero();
                particle.best_position = particle.position;
                particle.best_score = evaluateParticle(particle.position);

                if (particle.best_score > global_best_score_) {
                    global_best_score_ = particle.best_score;
                    global_best_position_ = particle.best_position;
                }
            }
        }

        void localOptimization() {
            typename LevenbergMarquardt<Scalar, Dim>::Options lm_options;
            lm_options.max_iterations = params_.local_search_iterations;

            auto result = lm_optimizer_.optimize(global_best_position_, error_func_, jacobian_func_);

            if (result) {
                VectorType optimized_position = result->position;
                Scalar optimized_score = evaluateParticle(optimized_position);

                if (optimized_score > global_best_score_) {
                    global_best_position_ = optimized_position;
                    global_best_score_ = optimized_score;
                }
            }
        }


        [[nodiscard]] bool isConverged() const {
            Scalar diversity = calculateSwarmDiversity();
            return diversity < params_.diversity_threshold;
        }
    };

} // namespace optimization

#endif // ADVANCED_PARTICLE_SWARM_OPTIMIZATION_HPP
