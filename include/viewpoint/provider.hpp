// File: viewpoint/provider.hpp

#ifndef VIEWPOINT_PROVIDER_HPP
#define VIEWPOINT_PROVIDER_HPP

#include <vector>
#include <memory>

#include "core/view.hpp"
#include "common/logging/logger.hpp"


namespace viewpoint {

    enum ProviderType {
        LOADER,
        GENERATOR
    };

    class Provider {
    public:
        virtual ~Provider() = default;

        // Function to provision viewpoints
        virtual std::vector<core::View> provision() = 0;

        virtual void setTargetImage(const cv::Mat &target_image) = 0;

        virtual void setCameraIntrinsics(const core::Camera::Intrinsics &camera_intrinsics) = 0;

        // Factory function to create a provider
        static std::unique_ptr<Provider> create(bool from_file, int num_samples, int dimensions);

    };

}

#endif // VIEWPOINT_PROVIDER_HPP

