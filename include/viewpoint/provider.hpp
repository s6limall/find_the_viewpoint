// File: viewpoint/provider.hpp

#ifndef VIEWPOINT_PROVIDER_HPP
#define VIEWPOINT_PROVIDER_HPP

#include <vector>
#include <memory>

#include <spdlog/spdlog.h>

#include "core/view.hpp"

namespace viewpoint {

    class Provider {
    public:
        virtual ~Provider() = default;

        // Function to provision viewpoints
        virtual std::vector<core::View> provision() = 0;

        // Factory function to create a provider
        static std::unique_ptr<Provider> createProvider(bool from_file, const std::string &path, int num_samples,
                                                        int dimension);
    };

}

#endif // VIEWPOINT_PROVIDER_HPP

