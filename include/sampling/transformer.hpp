// File: sampling/transformer.hpp

#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include <vector>
#include <stdexcept>

namespace sampling {

    template<typename T>
    class Transformer {
    public:
        virtual ~Transformer() = default;

        /**
         * @brief Transforms the given sample.
         * @param sample The sample to be transformed.
         * @return The transformed sample.
         * @throws std::runtime_error if transformation fails.
         */
        virtual std::vector<T> transform(const std::vector<T> &sample) const = 0;

        /**
         * @brief Validates the given sample.
         * @param sample The sample to be validated.
         * @throws std::invalid_argument if the sample is invalid.
         */
        virtual void validate(const std::vector<T> &sample) const {
            if (sample.size() != 3) {
                throw std::invalid_argument("Sample must have exactly three dimensions.");
            }
            for (const auto &value: sample) {
                if (value < 0.0 || value > 1.0) {
                    throw std::invalid_argument("Sample values must be in the range [0, 1].");
                }
            }
        }
    };

}


#endif //TRANSFORMER_HPP
