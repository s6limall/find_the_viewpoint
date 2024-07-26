// File: sampling/sampler.hpp

#ifndef SAMPLING_SAMPLER_HPP
#define SAMPLING_SAMPLER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <vector>

template<typename T = double>
class Sampler {
public:
    using TransformFunction =
            std::function<Eigen::Matrix<T, Eigen::Dynamic, 1>(const Eigen::Matrix<T, Eigen::Dynamic, 1> &)>;

    Sampler(const std::vector<T> &lower_bounds, const std::vector<T> &upper_bounds) :
        lower_bounds_(lower_bounds), upper_bounds_(upper_bounds), dimensions_(lower_bounds.size()) {
        assert(lower_bounds.size() == upper_bounds.size() && "Lower bounds and upper bounds must have the same size.");
    }

    virtual ~Sampler() = default;

    virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> generate(size_t num_samples,
                                                                      TransformFunction transform = nullptr) = 0;

    T discrepancy() const {
        if (samples_.cols() == 0) {
            LOG_ERROR("No samples_ generated.");
            throw std::invalid_argument("No samples_ generated.");
        }
        return discrepancy(samples_);
    }

    T discrepancy(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &samples) const {
        size_t num_samples = samples.cols();
        if (num_samples == 0 || dimensions_ == 0) {
            throw std::invalid_argument("Samples cannot be empty.");
        }

        T max_discrepancy = 0.0;

        for (size_t i = 0; i < num_samples; ++i) {
            Eigen::Matrix<T, Eigen::Dynamic, 1> point = samples.col(i);
            T volume = point.prod();

            size_t count = 0;
            for (size_t j = 0; j < num_samples; ++j) {
                if ((samples.col(j).array() <= point.array()).all()) {
                    count++;
                }
            }

            T discrepancy = std::abs(static_cast<T>(count) / num_samples - volume);
            max_discrepancy = std::max(max_discrepancy, discrepancy);
        }

        return max_discrepancy;
    }

protected:
    std::vector<T> lower_bounds_;
    std::vector<T> upper_bounds_;
    size_t dimensions_;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> samples_;

    Eigen::Matrix<T, Eigen::Dynamic, 1> mapToBounds(const Eigen::Matrix<T, Eigen::Dynamic, 1> &point) const {
        Eigen::Matrix<T, Eigen::Dynamic, 1> mapped_point(dimensions_);
        for (size_t i = 0; i < dimensions_; ++i) {
            mapped_point(i) = lower_bounds_[i] + point(i) * (upper_bounds_[i] - lower_bounds_[i]);
        }
        return mapped_point;
    }
};
#endif // SAMPLING_SAMPLER_HPP
