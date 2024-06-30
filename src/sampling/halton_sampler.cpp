// File: sampling/halton_sampler.cpp

#include "sampling/halton_sampler.hpp"

namespace sampling {
    std::vector<std::vector<double> > HaltonSampler::generate(
            const int num_samples,
            const std::vector<double> &lower_bounds,
            const std::vector<double> &upper_bounds) {
        if (num_samples <= 0) {
            throw std::invalid_argument("Number of samples must be positive.");
        }
        lower_bounds_ = lower_bounds;
        upper_bounds_ = upper_bounds;

        validateBounds();
        untouched_samples_.clear();
        samples_.clear();


        for (int i = 0; i < num_samples; ++i) {
            next(); // adds and adapts
        }

        LOG_DEBUG("Discrepancy before adaptation: {}", calculateDiscrepancy(untouched_samples_));
        LOG_DEBUG("Disrepancy after adaptation: {}", calculateDiscrepancy(samples_));

        return samples_;
    }

    std::vector<double> HaltonSampler::next() {
        if (lower_bounds_.empty() || upper_bounds_.empty()) {
            throw std::runtime_error("Sampler not initialized. Call generate() first.");
        }

        // Calculate the next sample using the current size of samples_ as the index.
        const int index = samples_.size();
        auto sample = scale(index);

        untouched_samples_.emplace_back(sample); // For debugging TODO: Remove

        if (adaptive_) {
            adapt(sample);
        }

        samples_.emplace_back(sample);

        return sample;
    }

    void HaltonSampler::adapt(std::vector<double> &sample) {

        // Default adaptation function to slightly perturb each sample point within the bounds.
        auto default_adapt = [this](std::vector<double> &point) {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution distribution(-0.01, 0.01);

            // Apply perturbation to each dimension of the sample point.
            for (int i = 0; i < point.size(); ++i) {
                const double perturbation = distribution(generator);
                point[i] = std::clamp(point[i] + perturbation * (upper_bounds_[i] - lower_bounds_[i]),
                                      lower_bounds_[i], upper_bounds_[i]);
            }
        };

        const auto &adapt_function = custom_adapt_ ? custom_adapt_ : default_adapt;

        LOG_TRACE("Using {} adaptation function.", custom_adapt_ ? "custom" : "default (perturbator)");
        adapt_function(sample);
    }

    int HaltonSampler::nextPrime(const int after) {
        // Find the next prime number greater than the given number.
        for (int num = after + 1;; ++num) {
            if (std::all_of(primes_.begin(), primes_.end(), [num](const int div) { return num % div != 0; })) {
                return num;
            }
        }
    }

    double HaltonSampler::halton(int index, const int base) noexcept {
        double result = 0.0;
        double fraction = 1.0;

        // Compute Halton sequence value using the given base.
        while (index > 0) {
            fraction /= base;
            result += fraction * (index % base);
            index /= base;
        }
        return result;
    }


    std::vector<double> HaltonSampler::scale(int index) const {
        LOG_TRACE("Scaling sample at index {}.", index);
        const int dimension = lower_bounds_.size();
        std::vector<double> sample(dimension);

        // Scale the Halton sequence value to the specified bounds for each dimension.
        for (int j = 0; j < dimension; ++j) {
            const int base = (j < primes_.size()) ? primes_[j] : nextPrime(primes_.back());
            const double halton_value = halton(index + 1, base);
            sample[j] = lower_bounds_[j] + halton_value * (upper_bounds_[j] - lower_bounds_[j]);
        }

        return sample;
    }
}
