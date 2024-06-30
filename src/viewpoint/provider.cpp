// File: viewpoint/provider.cpp

#include "viewpoint/provider.hpp"

#include "config/configuration.hpp"
#include "common/logging/logger.hpp"
#include "viewpoint/loader.hpp"
#include "viewpoint/generator.hpp"

namespace viewpoint {

    std::unique_ptr<Provider> Provider::create(const bool from_file, const int num_samples,
                                               const int dimensions) {
        if (from_file) {
            std::string filepath = config::get("paths.view_space_file", "../view_space/5.txt");
            LOG_INFO("Creating a Viewpoint Loader to load viewpoints from file: {}", filepath);
            return std::make_unique<Loader>(filepath);
        } else {
            LOG_INFO("Creating a Viewpoint Generator to generate {} viewpoints with {} dimensions.", num_samples,
                     dimensions);
            return std::make_unique<Generator>(num_samples, dimensions);
        }
    }

}
