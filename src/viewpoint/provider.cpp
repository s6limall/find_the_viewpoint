// File: viewpoint/provider.cpp

#include "viewpoint/provider.hpp"
#include "viewpoint/loader.hpp"
#include "viewpoint/generator.hpp"

namespace viewpoint {

    std::unique_ptr<Provider> Provider::createProvider(bool from_file, const std::string &filepath, int num_samples,
                                                       int dimensions) {
        if (from_file) {
            spdlog::info("Creating Loader to load viewpoints from file: {}", filepath);
            return std::make_unique<Loader>(filepath);
        } else {
            spdlog::info("Creating Generator to generate {} viewpoints with {} dimensions", num_samples, dimensions);
            return std::make_unique<Generator>(num_samples, dimensions);
        }
    }

}
