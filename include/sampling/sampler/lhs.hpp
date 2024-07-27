// File: sampling/sampler/lhs.hpp

#ifndef SAMPLING_LHS_SAMPLER_HPP
#define SAMPLING_LHS_SAMPLER_HPP

#include <Eigen/Dense>
#include <random>
#include <stdexcept>

#include "common/logging/logger.hpp"
#include "sampling/sampler.hpp"


template<typename T = double>
class LHSSampler final : public Sampler<T> {
public:
    using typename Sampler<T>::TransformFunction;

    LHSSampler(const std::vector<T> &lower_bounds, const std::vector<T> &upper_bounds) :
        Sampler<T>(lower_bounds, upper_bounds), rng_(std::random_device{}()) {
        if (lower_bounds.size() != upper_bounds.size()) {
            LOG_ERROR("Invalid bounds: Lower bounds and upper bounds must have the same size.");
            throw std::invalid_argument("Lower bounds and upper bounds must have the same size.");
        }
        LOG_DEBUG("Latin Hypercube Sampler initialized with dimensions: {}", this->dimensions_);
        for (size_t i = 0; i < lower_bounds.size(); ++i) {
            if (lower_bounds[i] >= upper_bounds[i]) {
                LOG_ERROR("Invalid bounds: At index {}, lower bound {} is greater than or equal to upper bound {}.", i,
                          lower_bounds[i], upper_bounds[i]);
                throw std::invalid_argument("Each lower bound must be less than the corresponding upper bound.");
            }
        }
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> generate(size_t num_samples,
                                                              TransformFunction transform = nullptr) override {
        if (num_samples == 0) {
            LOG_ERROR("Invalid number of samples: Number of samples must be greater than zero.");
            throw std::invalid_argument("Number of samples must be greater than zero.");
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> samples(this->dimensions_, num_samples);
        std::uniform_real_distribution<T> dist(0.0, 1.0);

        LOG_DEBUG("Generating {} samples.", num_samples);
        for (size_t i = 0; i < this->dimensions_; ++i) {
            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::ranges::shuffle(indices.begin(), indices.end(), rng_);

            for (size_t j = 0; j < num_samples; ++j) {
                T interval_start = static_cast<T>(indices[j]) / num_samples;
                T interval_end = static_cast<T>(indices[j] + 1) / num_samples;
                samples(i, j) = dist(rng_) * (interval_end - interval_start) + interval_start;
            }
        }
        LOG_DEBUG("Sampling complete.");

        LOG_DEBUG("Mapping samples to bounds.");
        for (size_t j = 0; j < num_samples; ++j) {
            Eigen::Matrix<T, Eigen::Dynamic, 1> sample = samples.col(j);
            samples.col(j) = this->mapToBounds(sample);
        }
        LOG_DEBUG("Bounds mapping complete.");

        if (transform) {
            LOG_DEBUG("Transforming samples.");
            for (size_t j = 0; j < num_samples; ++j) {
                samples.col(j) = transform(samples.col(j));
            }
            LOG_DEBUG("Transformation complete.");
        }

        this->samples_ = samples;

        return this->samples_;
    }

private:
    std::mt19937 rng_;
};


#endif // SAMPLING_LHS_SAMPLER_HPP
