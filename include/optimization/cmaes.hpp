// File: optimization/cmaes.hpp

#ifndef CMAES_HPP
#define CMAES_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "sampling/transformer.hpp"

namespace optimization {

    template<typename T = double>
    class CMAES {
    public:
        struct Parameters {
            int population_size;
            int num_dimensions;
            T initial_step_size;
            T termination_tol;
            int max_iterations;
        };

        struct Solution {
            Eigen::Matrix<T, Eigen::Dynamic, 1> x;
            T fitness;
        };

        explicit CMAES(const Parameters &params);

        void initialize(const Eigen::Matrix<T, Eigen::Dynamic, 1> &initial_mean);
        void initialize(const std::vector<Solution> &initial_population);
        std::vector<Solution> samplePopulation(const std::shared_ptr<Transformer<T>> &transformer);
        void updatePopulation(const std::vector<Solution> &population);
        [[nodiscard]] bool terminationCriteria() const;

    private:
        Parameters params_;
        Eigen::Matrix<T, Eigen::Dynamic, 1> mean_;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> cov_;
        T step_size_;
        int iteration_;
        std::mt19937 rng_;
    };

    template<typename T>
    CMAES<T>::CMAES(const Parameters &params) :
        params_(params), step_size_(params.initial_step_size), iteration_(0), rng_(std::random_device{}()) {}

    template<typename T>
    void CMAES<T>::initialize(const Eigen::Matrix<T, Eigen::Dynamic, 1> &initial_mean) {
        mean_ = initial_mean;
        cov_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(params_.num_dimensions,
                                                                          params_.num_dimensions);
    }

    template<typename T>
    void CMAES<T>::initialize(const std::vector<Solution> &initial_population) {
        mean_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(params_.num_dimensions);
        for (const auto &sol: initial_population) {
            mean_ += sol.x;
        }
        mean_ /= initial_population.size();

        cov_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(params_.num_dimensions, params_.num_dimensions);
        for (const auto &sol: initial_population) {
            Eigen::Matrix<T, Eigen::Dynamic, 1> diff = sol.x - mean_;
            cov_ += diff * diff.transpose();
        }
        cov_ /= initial_population.size();
    }

    template<typename T>
    std::vector<typename CMAES<T>::Solution>
    CMAES<T>::samplePopulation(const std::shared_ptr<Transformer<T>> &transformer) {
        std::normal_distribution<T> dist(0.0, 1.0);
        std::vector<Solution> population(params_.population_size);

        for (int i = 0; i < params_.population_size; ++i) {
            Eigen::Matrix<T, Eigen::Dynamic, 1> z = Eigen::Matrix<T, Eigen::Dynamic, 1>::NullaryExpr(
                    params_.num_dimensions, [&](int) { return dist(rng_); });
            Eigen::Matrix<T, Eigen::Dynamic, 1> y = mean_ + step_size_ * cov_ * z;

            // Ensure samples are within bounds [0, 1]
            y = y.cwiseMax(0.0).cwiseMin(1.0);

            population[i] = {transformer->transform(y), 0.0}; // Fitness will be evaluated later
        }
        return population;
    }

    template<typename T>
    void CMAES<T>::updatePopulation(const std::vector<Solution> &population) {
        std::vector<T> weights(params_.population_size);
        T sum_weights = 0.0;
        for (int i = 0; i < params_.population_size; ++i) {
            weights[i] = std::log(params_.population_size + 0.5) - std::log(i + 1);
            sum_weights += weights[i];
        }
        for (auto &w: weights) {
            w /= sum_weights;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> new_mean =
                Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(params_.num_dimensions);
        for (int i = 0; i < params_.population_size; ++i) {
            new_mean += weights[i] * population[i].x;
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> new_cov =
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(params_.num_dimensions, params_.num_dimensions);
        for (int i = 0; i < params_.population_size; ++i) {
            Eigen::Matrix<T, Eigen::Dynamic, 1> y = (population[i].x - mean_) / step_size_;
            new_cov += weights[i] * y * y.transpose();
        }
        cov_ = new_cov;
        mean_ = new_mean;

        step_size_ *= std::exp((1.0 / params_.num_dimensions) *
                               (population[0].fitness - population[params_.population_size / 2].fitness) /
                               params_.termination_tol);
        ++iteration_;
    }

    template<typename T>
    bool CMAES<T>::terminationCriteria() const {
        return iteration_ >= params_.max_iterations;
    }

} // namespace optimization


#endif // CMAES_HPP
