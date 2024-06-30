// File: sampling/sampler.hpp

#ifndef SAMPLING_SAMPLER_HPP
#define SAMPLING_SAMPLER_HPP

#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>

#include "common/logging/logger.hpp"

namespace sampling {
    class Sampler {
    public:
        virtual ~Sampler() = default;

        /**
         * @brief Generates a specified number of samples within the given bounds.
         * @param num_samples Number of samples to generate.
         * @param lower_bounds Lower bounds of the sample space.
         * @param upper_bounds Upper bounds of the sample space.
         * @return A vector of generated samples.
         */
        virtual std::vector<std::vector<double> > generate(
                int num_samples,
                const std::vector<double> &lower_bounds,
                const std::vector<double> &upper_bounds) = 0;

        /**
         * @brief Generates the next sample in the sequence.
         * @return The next generated sample.
         */
        virtual std::vector<double> next() = 0;

        /**
         * @brief Resets the sampler to its initial state.
         */
        void reset();

        /**
         * @brief Sets the adaptive mode with an optional custom adaptation function.
         * @param adaptive Enable or disable adaptive mode.
         * @param custom_adapt Custom adaptation function.
         * @return Reference to the Sampler instance.
         */
        Sampler &setAdaptive(bool adaptive,
                             std::function<void(std::vector<double> &)> custom_adapt = nullptr) noexcept;


        /**
         * @brief Calculates the star discrepancy of the generated samples.
         * @return The star discrepancy.
         */
        [[nodiscard]] static double calculateDiscrepancy(std::vector<std::vector<double> > samples);

        [[nodiscard]] double discrepancy() const;

    protected:
        std::vector<double> lower_bounds_;
        std::vector<double> upper_bounds_;
        std::vector<std::vector<double> > samples_;

        bool adaptive_ = false;
        std::function<void(std::vector<double> &)> custom_adapt_;

        virtual void adapt(std::vector<double> &sample) = 0;

        /**
         * @brief Validates that the bounds are correct.
         * @throws std::invalid_argument if bounds are incorrect.
         */
        void validateBounds() const;
    };
}

#endif // SAMPLING_SAMPLER_HPP
