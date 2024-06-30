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


        virtual void setTargetImage(const cv::Mat &target_image) = 0;

        virtual void setCameraParameters(const core::Camera::CameraParameters &camera_parameters) = 0;

        // Factory function to create a provider
        static std::unique_ptr<Provider> create(bool from_file, int num_samples, int dimensions);
    };

}

#endif // VIEWPOINT_PROVIDER_HPP

