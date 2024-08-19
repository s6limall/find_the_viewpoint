// File: core/perception.hpp

#ifndef PERCEPTION_HPP
#define PERCEPTION_HPP

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include "core/camera.hpp"

namespace core {

    class Perception {
    public:
        virtual ~Perception() = default;

        // Render the object using the given camera pose and save the image.
        [[nodiscard]] virtual cv::Mat render(const Eigen::Matrix4d &extrinsics, std::string_view save_path) = 0;

        // Provide camera to be used in views
        [[nodiscard]] std::shared_ptr<Camera> getCamera() { return camera_; }

    protected:
        std::shared_ptr<Camera> camera_;
        Perception() = default;
    };

} // namespace core

#endif // PERCEPTION_HPP
