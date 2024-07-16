// File: sampling/sampler.hpp

#ifndef SAMPLING_SAMPLER_HPP
#define SAMPLING_SAMPLER_HPP

#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <string_view>
#include <cmath>
#include <optional>

#include "transformer.hpp"
#include "common/logging/logger.hpp"

namespace sampling {

    template<typename T>
    using Adaptation = std::function<void(std::vector<T> &)>;
    template<typename T>
    using Transformation = std::optional<std::function<std::vector<T>(const std::vector<T> &)> >;
    template<typename T>
    using Points = std::vector<std::vector<T> >;

    class Sampler {
    public:
        explicit Sampler(const Transformation<double> &transform = std::nullopt);

        // TODO: std::optional cannot hold abstract types
        // explicit Sampler(const std::optional<Transformer<double> > &transformer = std::nullopt);

        virtual ~Sampler() = default;

        /**
         * @brief Generates a specified number of samples within the given bounds.
         * @param num_samples Number of samples to generate.
         * @param lower_bounds Lower bounds of the sample space.
         * @param upper_bounds Upper bounds of the sample space.
         * @return A vector of generated samples.
         */
        virtual Points<double> generate(
                size_t num_samples,
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
        void reset() noexcept;

        /**
         * @brief Sets the adaptive mode with an optional custom adaptation function.
         * @param adaptive Enable or disable adaptive mode.
         * @param adapt Custom adaptation function.
         * @return Reference to the Sampler instance.
         */
        Sampler &setAdaptive(bool adaptive, Adaptation<double> adapt = nullptr) noexcept;

        /**
         * @brief Calculates the star discrepancy of the generated samples.
         * @return The star discrepancy.
         */
        [[nodiscard]] static double calculateDiscrepancy(const Points<double> &samples);

        [[nodiscard]] double discrepancy() const;

    protected:
        bool adaptive_ = false;
        std::vector<double> lower_bounds_;
        std::vector<double> upper_bounds_;
        Points<double> samples_;
        size_t current_index_ = 0;

        Transformation<double> transform_;
        Adaptation<double> adapt_;

        virtual void adapt(std::vector<double> &sample) = 0;

        /**
         * @brief Validates that the bounds are correct.
         * @throws std::invalid_argument if bounds are incorrect.
         */
        void validateBounds() const;
    };

}

#endif // SAMPLING_SAMPLER_HPP
