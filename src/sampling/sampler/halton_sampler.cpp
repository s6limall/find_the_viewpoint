// File: sampling/halton_sampler.cpp

#include "sampling/sampler/halton_sampler.hpp"

namespace sampling {


    Points<double> HaltonSampler::generate(
            const std::size_t num_samples,
            const std::vector<double> &lower_bounds,
            const std::vector<double> &upper_bounds) {

        if (num_samples == 0) {
            throw std::invalid_argument("Number of samples must be positive.");
        }

        LOG_INFO("Sampling using Halton Sequences: adaptive = {}, transform = {}.", adaptive_, transform_ != nullptr);
        LOG_INFO("Generating {} Halton samples in the range {} to {}.", num_samples, lower_bounds, upper_bounds);

        lower_bounds_ = lower_bounds;
        upper_bounds_ = upper_bounds;

        validateBounds();
        untouched_samples_.clear();
        samples_.clear();

        for (std::size_t i = 0; i < num_samples; ++i) {
            auto sample = next();
            LOG_TRACE("Generated sample (untransformed): {}", sample);
            if (transform_) {
                sample = (*transform_)(sample);
                LOG_TRACE("Transformed sample: {}", sample);
            }
            samples_.emplace_back(sample);
        }


        LOG_DEBUG("Discrepancy before adaptation: {}", calculateDiscrepancy(untouched_samples_));
        LOG_DEBUG("Discrepancy after adaptation: {}", calculateDiscrepancy(samples_));

        return samples_;
    }

    std::vector<double> HaltonSampler::next() {
        if (lower_bounds_.empty() || upper_bounds_.empty()) {
            LOG_ERROR("Halton Sampler not initialized, bounds not set. Call generate() first.");
            throw std::runtime_error("Sampler not initialized. Call generate() first.");
        }

        const auto index = samples_.size();
        auto sample = scale(index);

        untouched_samples_.emplace_back(sample); // For debugging TODO: Remove later

        if (adaptive_) {
            LOG_TRACE("Before adaptation: {}", sample);
            adapt(sample);
            LOG_TRACE("After adaptation: {}", sample);
        }

        LOG_TRACE("Scaled sample: {}", sample);

        return sample;
    }

    void HaltonSampler::adapt(std::vector<double> &sample) {
        auto default_adapt = [this](std::vector<double> &point) {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution<double> distribution(-0.01, 0.01);

            for (std::size_t i = 0; i < point.size(); ++i) {
                const double perturbation = distribution(generator);
                point[i] = std::clamp(point[i] + perturbation * (upper_bounds_[i] - lower_bounds_[i]),
                                      lower_bounds_[i], upper_bounds_[i]);
            }
        };

        const auto &adapt_function = adapt_ ? adapt_ : default_adapt;

        LOG_TRACE("Using {} adaptation function.", adapt_ ? "custom" : "default (perturbator)");
        adapt_function(sample);
    }

    int HaltonSampler::nextPrime(const int after) noexcept {
        auto isPrime = [](int num) {
            return std::all_of(primes_.begin(), primes_.end(), [num](const int div) { return num % div != 0; });
        };

        for (int num = after + 1;; ++num) {
            if (isPrime(num)) {
                return num;
            }
        }

    }

    double HaltonSampler::halton(int index, const int base) noexcept {
        double result = 0.0;
        double fraction = 1.0;

        while (index > 0) {
            fraction /= base;
            result += fraction * (index % base);
            index /= base;
        }

        LOG_TRACE("Halton Sample at index {} with base {} is {}", index, base, result);

        return result;
    }

    std::vector<double> HaltonSampler::scale(size_t index) const {
        LOG_TRACE("Scaling sample at index {}.", index);
        const auto dimension = lower_bounds_.size();
        std::vector<double> sample(dimension);

        for (std::size_t j = 0; j < dimension; ++j) {
            const int base = (j < primes_.size()) ? primes_[j] : nextPrime(primes_.back());
            const double halton_value = halton(static_cast<int>(index) + 1, base);
            sample[j] = lower_bounds_[j] + halton_value * (upper_bounds_[j] - lower_bounds_[j]);
        }

        return sample;
    }


}
