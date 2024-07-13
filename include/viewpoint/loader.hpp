// File: viewpoint/loader.hpp

#ifndef VIEWPOINT_LOADER_HPP
#define VIEWPOINT_LOADER_HPP


#include "viewpoint/provider.hpp"
#include "common/io/io.hpp"

namespace viewpoint {

    class Loader final : public Provider {
    public:
        explicit Loader(std::string filepath);

        std::vector<core::View> provision() override;

        void setTargetImage(const cv::Mat &target_image) override {
        };

        void setCameraIntrinsics(const core::Camera::Intrinsics &camera_intrinsics) override {
        };

    private:
        const std::string filepath_;
    };

}

#endif // VIEWPOINT_LOADER_HPP

