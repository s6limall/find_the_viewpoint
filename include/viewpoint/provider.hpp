// File: viewpoint/provider.hpp

#ifndef VIEWPOINT_PROVIDER_HPP
#define VIEWPOINT_PROVIDER_HPP

#include <vector>
#include <memory>

#include "core/view.hpp"
#include "common/logging/logger.hpp"
#include "types/viewpoint.hpp"


namespace viewpoint {

    enum ProviderType {
        LOADER,
        GENERATOR
    };

    template<typename T = double> // generator, loader
    class Provider {

    public:
        virtual ~Provider() = default;

        // Function to provision viewpoints
        virtual std::vector<ViewPoint<T> > provision() = 0;
    };

}

#endif // VIEWPOINT_PROVIDER_HPP
