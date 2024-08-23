#ifndef ROBOT_HPP
#define ROBOT_HPP

#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>
#include "core/camera.hpp"
#include "core/vision/perception.hpp"

namespace core {

    class Robot final : public Perception {
    public:
        using MoveCallback = std::function<void(const Eigen::Matrix4d &)>;
        using CaptureCallback = std::function<cv::Mat()>;

        Robot(MoveCallback move_cb, CaptureCallback capture_cb) :
            move_callback_(std::move(move_cb)), capture_callback_(std::move(capture_cb)) {}

        cv::Mat render(const Eigen::Matrix4d &extrinsics, const std::string_view save_path) override {
            // Move the robot to the desired position
            move_callback_(extrinsics);

            // Capture an image using the provided callback
            cv::Mat captured_image = capture_callback_();

            // Save the image if a save path is provided
            if (!save_path.empty()) {
                cv::imwrite(std::string(save_path), captured_image);
            }

            return captured_image;
        }

    private:
        MoveCallback move_callback_;
        CaptureCallback capture_callback_;
    };

} // namespace core

#endif // ROBOT_HPP
