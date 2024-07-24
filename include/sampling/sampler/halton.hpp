// File: sampling/halton.hpp

#ifndef HALTON_SAMPLER_HPP
#define HALTON_SAMPLER_HPP


#include <cassert>
#include <cmath>
#include <vector>
#include "common/logging/logger.hpp"
#include "sampling/sampler.hpp"

template<typename T = double>
class HaltonSampler final : public Sampler<T> {
public:
    using TransformFunction = typename Sampler<T>::TransformFunction;

    HaltonSampler(const std::vector<T> &lower_bounds, const std::vector<T> &upper_bounds) :
        Sampler<T>(lower_bounds, upper_bounds), bases_(generatePrimes(this->dimensions_)) {
        LOG_INFO("HaltonSampler initialized with dimensions: {}", this->dimensions_);
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> generate(size_t num_samples,
                                                              TransformFunction transform = nullptr) override {
        this->samples.resize(this->dimensions_, num_samples);
        fillSamples(this->samples, num_samples, transform);
        LOG_INFO("Generated {} samples.", num_samples);
        LOG_DEBUG("Discrepancy: {}", this->discrepancy(this->samples));
        return this->samples;
    }

private:
    std::vector<int> bases_;

    static std::vector<int> generatePrimes(size_t count) {
        std::vector<int> primes;
        primes.reserve(count);
        int number = 2;
        while (primes.size() < count) {
            if (isPrime(number)) {
                primes.push_back(number);
            }
            ++number;
        }
        LOG_INFO("Generated {} primes for bases: {}", count, primes);
        return primes;
    }

    static constexpr bool isPrime(const int number) noexcept {
        if (number < 2)
            return false;
        for (int divisor = 2; divisor * divisor <= number; ++divisor) {
            if (number % divisor == 0)
                return false;
        }
        return true;
    }

    static constexpr T haltonSequence(int index, int base) noexcept {
        T result = 0.0;
        T f = 1.0 / static_cast<T>(base);
        while (index > 0) {
            result += f * (index % base);
            index /= base;
            f /= base;
        }
        return result;
    }

    void fillSamples(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &samples, const size_t num_samples,
                     TransformFunction transform) const {
        for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
            auto point = generatePoint(static_cast<int>(sample_index) + 1);
            if (transform) {
                point = transform(point);
            }
            samples.col(sample_index) = this->mapToBounds(point);
        }
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> generatePoint(const int index) const {
        Eigen::Matrix<T, Eigen::Dynamic, 1> point(this->dimensions_);
        for (size_t dimension_index = 0; dimension_index < this->dimensions_; ++dimension_index) {
            point(dimension_index) = haltonSequence(index, bases_[dimension_index]);
        }
        LOG_DEBUG("Point: ({}, {}, {})", point(0), point(1), point(2));
        return point;
    }
};


#endif // HALTON_SAMPLER_HPP
