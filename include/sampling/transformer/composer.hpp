// File: sampling/transformer/composer.hpp

#ifndef TRANSFORMER_COMPOSER_HPP
#define TRANSFORMER_COMPOSER_HPP

#include <memory>
#include "sampling/transformer.hpp"

template<typename T = double>
class TransformerComposer final : public Transformer<T> {
public:
    explicit TransformerComposer(std::vector<std::shared_ptr<Transformer<T>>> transformers) :
        transformers_(std::move(transformers)) {}

    Eigen::Matrix<T, Eigen::Dynamic, 1> transform(const Eigen::Matrix<T, Eigen::Dynamic, 1> &sample) const override {
        Eigen::Matrix<T, Eigen::Dynamic, 1> result = sample;
        for (const auto &transformer: transformers_) {
            result = transformer->transform(result);
        }
        return result;
    }

    void validate(const Eigen::Matrix<T, Eigen::Dynamic, 1> &sample) const override {
        for (const auto &transformer: transformers_) {
            transformer->validate(sample);
        }
    }

private:
    std::vector<std::shared_ptr<Transformer<T>>> transformers_;
};

#endif // TRANSFORMER_COMPOSER_HPP
