// File: viewpoint/provider.hpp

#ifndef PROVIDER_HPP
#define PROVIDER_HPP

#include <vector>
#include "types/image.hpp"

namespace viewpoint {

    // Interface
    template<typename T = double>
    class Provider {
    public:
        virtual ~Provider() = default;
        virtual std::vector<Image<T>> provision(size_t num_points) = 0;
        virtual Image<T> next() = 0;
    };
} // namespace viewpoint

#endif // PROVIDER_HPP
