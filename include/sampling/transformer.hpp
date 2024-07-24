// File: sampling/transformer.hpp

#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <vector>

template<typename T = double>
class Transformer {
public:
    virtual ~Transformer() = default;

    /**
     * @brief Transforms the given sample.
     * @param sample The sample to be transformed.
     * @return The transformed sample.
     * @throws std::runtime_error if transformation fails.
     */
    virtual Eigen::Matrix<T, Eigen::Dynamic, 1> transform(const Eigen::Matrix<T, Eigen::Dynamic, 1> &sample) const = 0;

    /**
     * @brief Validates the given sample.
     * @param sample The sample to be validated.
     * @throws std::invalid_argument if the sample is invalid.
     */
    virtual void validate(const Eigen::Matrix<T, Eigen::Dynamic, 1> &sample) const {
        if (sample.size() < 1) {
            LOG_ERROR("Sample must have at least one dimension. Received: {}", sample.size());
            throw std::invalid_argument("Sample must have at least one dimension.");
        }
        LOG_DEBUG("Sample ({}, {}, {}) validated successfully.", sample.x(), sample.y(), sample.z());
    }
};

#endif // TRANSFORMER_HPP
