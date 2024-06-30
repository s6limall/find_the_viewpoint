// File: sampling/sampler.cpp

#include "sampling/sampler.hpp"

namespace sampling {

    Sampler &Sampler::setAdaptive(const bool adaptive,
                                  std::function<void(std::vector<double> &)> custom_adapt) noexcept {
        LOG_TRACE("Setting adaptive mode to {}.", adaptive);
        adaptive_ = adaptive;
        custom_adapt_ = std::move(custom_adapt);
        return *this;
    }

    void Sampler::reset() {
        LOG_TRACE("Resetting sampler, clearing {} samples.", samples_.size());
        samples_.clear();
    }

    void Sampler::validateBounds() const {
        LOG_TRACE("Validating bounds...");
        if (lower_bounds_.empty() || upper_bounds_.empty()) {
            throw std::invalid_argument("Bounds cannot be empty.");
        }

        if (lower_bounds_.size() != upper_bounds_.size()) {
            throw std::invalid_argument("Lower and upper bounds must have the same dimension.");
        }

        if (!std::equal(lower_bounds_.begin(), lower_bounds_.end(), upper_bounds_.begin(), std::less<>())) {
            throw std::invalid_argument("Each lower bound must be less than the corresponding upper bound.");
        }

        LOG_TRACE("Bounds validated! Upper bounds: {}, Lower bounds: {}", upper_bounds_, lower_bounds_);
    }

    double Sampler::discrepancy() const {
        return calculateDiscrepancy(this->samples_);
    }

    double Sampler::calculateDiscrepancy(std::vector<std::vector<double> > samples) {
        if (samples.empty() || samples[0].empty()) {
            throw std::invalid_argument("Samples cannot be empty.");
        }

        const size_t num_samples = samples.size();

        double max_discrepancy = 0.0;

        // Iterate over all points to calculate the star discrepancy
        for (const auto &point: samples) {
            const double volume = std::accumulate(point.begin(), point.end(), 1.0, std::multiplies<>());

            const size_t count = std::count_if(samples.begin(), samples.end(),
                                               [&point](const std::vector<double> &sample) {
                                                   return std::inner_product(
                                                           sample.begin(), sample.end(), point.begin(), true,
                                                           std::logical_and<>(),
                                                           std::less_equal<>());
                                               });

            const double term = static_cast<double>(count) / static_cast<double>(num_samples);
            max_discrepancy = std::max(max_discrepancy, std::abs(term - volume));
        }

        return max_discrepancy;
    }

}
