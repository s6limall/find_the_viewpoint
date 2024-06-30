// File: viewpoint/loader.hpp

#ifndef VIEWPOINT_LOADER_HPP
#define VIEWPOINT_LOADER_HPP

#include "viewpoint/provider.hpp"
#include <string>
#include <vector>
#include "core/view.hpp"

namespace viewpoint {

    class Loader final : public Provider {
    public:
        explicit Loader(std::string filepath);

        std::vector<core::View> provision() override;

        void setTargetImage(const cv::Mat &target_image) override {
        };

        void setCameraParameters(const core::Camera::CameraParameters &camera_parameters) override {
        };

    private:
        std::string filepath_;
    };

}

#endif // VIEWPOINT_LOADER_HPP

