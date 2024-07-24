// File: sampling/transformer/builder/builder.hpp

#ifndef TRANSFORMER_BUILDER_HPP
#define TRANSFORMER_BUILDER_HPP

#include <memory>
#include <vector>

#include "sampling/transformer.hpp"
#include "sampling/transformer/composer.hpp"

template<typename T = double>
class TransformerBuilder {
public:
    TransformerBuilder &add(std::shared_ptr<Transformer<T>> transformer) {
        transformers_.push_back(transformer);
        return *this;
    }

    std::shared_ptr<Transformer<T>> build() { return std::make_shared<TransformerComposer<T>>(transformers_); }

private:
    std::vector<std::shared_ptr<Transformer<T>>> transformers_;
};


#endif // TRANSFORMER_BUILDER_HPP
