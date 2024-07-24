// File: sampling/normalizing_transformer.hpp

#ifndef NORMALIZING_TRANSFORMER_HPP
#define NORMALIZING_TRANSFORMER_HPP

#include "sampling/transformer.hpp"

template<typename T = double>
class NormalizingTransformer final : public Transformer<T> {
public:
    NormalizingTransformer(std::initializer_list<T> lower_bounds, std::initializer_list<T> upper_bounds) :
        lower_bounds_(lower_bounds), upper_bounds_(upper_bounds) {
        if (lower_bounds_.size() != upper_bounds_.size()) {
            LOG_ERROR("Lower and upper bounds must have the same number of dimensions. Received: {} and {}.",
                      lower_bounds_.size(), upper_bounds_.size());
            throw std::invalid_argument("Lower and upper bounds must have the same number of dimensions.");
        }
        LOG_DEBUG("Normalizing transformer initialized with {} dimensions.", lower_bounds_.size());
    }

    NormalizingTransformer(const std::vector<T> &lower_bounds, const std::vector<T> &upper_bounds) :
        lower_bounds_(lower_bounds), upper_bounds_(upper_bounds) {
        if (lower_bounds_.size() != upper_bounds_.size()) {
            LOG_ERROR("Lower and upper bounds must have the same number of dimensions. Received: {} and {}.",
                      lower_bounds_.size(), upper_bounds_.size());
            throw std::invalid_argument("Lower and upper bounds must have the same number of dimensions.");
        }
        LOG_DEBUG("Normalizing transformer initialized with {} dimensions.", lower_bounds_.size());
    }


    Eigen::Matrix<T, Eigen::Dynamic, 1> transform(const Eigen::Matrix<T, Eigen::Dynamic, 1> &sample) const override {
        this->validate(sample);

        T min_value = sample.minCoeff();
        T max_value = sample.maxCoeff();
        LOG_DEBUG("Sample min: {}, max: {}.", min_value, max_value);

        if (min_value == max_value) {
            LOG_ERROR("Input sample has zero variance. Sample: ({}, {}, {})", sample(0), sample(1), sample(2));
            throw std::runtime_error("Input sample has zero variance.");
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> normalized(sample.size());
        for (int i = 0; i < sample.size(); ++i) {
            normalized(i) = lower_bounds_[i] +
                            (upper_bounds_[i] - lower_bounds_[i]) * ((sample(i) - min_value) / (max_value - min_value));
        }
        LOG_DEBUG("Sample ({}, {}, {}) normalized to ({}, {}, {}).", sample(0), sample(1), sample(2), normalized(0),
                  normalized(1), normalized(2));

        if (normalized.hasNaN()) {
            LOG_ERROR("NormalizingTransformer produced NaN values. Sample: ({}, {}, {}), Transformed: ({}, {}, {})",
                      sample(0), sample(1), sample(2), normalized(0), normalized(1), normalized(2));
            throw std::runtime_error("NormalizingTransformer produced NaN values.");
        }

        LOG_DEBUG("Sample normalized to ({}, {}, {}).", normalized(0), normalized(1), normalized(2));

        return normalized;
    }

protected:
    void validate(const Eigen::Matrix<T, Eigen::Dynamic, 1> &sample) const override {
        Transformer<T>::validate(sample);
        if (sample.size() != lower_bounds_.size()) {
            LOG_ERROR("Sample must have the same number of dimensions as bounds. Received: {} and {}.", sample.size(),
                      lower_bounds_.size());
            throw std::invalid_argument("Sample must have the same number of dimensions as bounds.");
        }
        LOG_DEBUG("Sample ({}, {}, {}) validated successfully.", sample.x(), sample.y(), sample.z());
    }

private:
    std::vector<T> lower_bounds_;
    std::vector<T> upper_bounds_;
};

#endif // NORMALIZING_TRANSFORMER_HPP
