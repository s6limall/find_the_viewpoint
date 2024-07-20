// File: viewpoint/provider.cpp

#include "viewpoint/provider.hpp"


/*
// File: viewpoint/provider.cpp

#include "viewpoint/provider.hpp"

#include "common/logging/logger.hpp"
#include "config/configuration.hpp"
#include "viewpoint/generator.hpp"
#include "viewpoint/loader.hpp"

namespace viewpoint {

    std::unique_ptr<Provider> Provider::create(const bool from_file, const int num_samples,
                                               const int dimensions) {
        if (from_file) {
            auto file_path = config::get("paths.view_space_file", "../view_space/5.txt");
            LOG_DEBUG("Creating a Viewpoint Loader to load viewpoints from file '{}'.", file_path);
            return std::make_unique<Loader>(file_path);
        }

        LOG_DEBUG("Creating a Viewpoint Generator to generate {} points with {} dimensions.", num_samples, dimensions);
        // return std::make_unique<Generator>(num_samples, dimensions);
    }
}
*/
