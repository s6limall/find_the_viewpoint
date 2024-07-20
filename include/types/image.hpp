// File: types/image.hpp

#ifndef TYPE_IMAGE_HPP
#define TYPE_IMAGE_HPP

#include <fmt/core.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <utility>
#include <vector>

#include "common/logging/logger.hpp"
#include "types/viewpoint.hpp"

using FeatureExtractor = processing::image::FeatureExtractor;

template<typename T = double>
class Image {
public:
    Image() noexcept = default;

    Image(cv::Mat image, const std::unique_ptr<FeatureExtractor> &extractor) : image_{validateImage(std::move(image))} {
        this->detect(extractor);
    }

    explicit Image(cv::Mat image, const cv::Ptr<cv::Feature2D> &detector = cv::ORB::create()) :
        image_{validateImage(std::move(image))} {
        this->detect(detector);
    }

    Image(cv::Mat image, cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints) :
        image_{validateImage(std::move(image))}, descriptors_{std::move(descriptors)},
        keypoints_{std::move(keypoints)} {}

    // Getters
    [[nodiscard]] const cv::Mat &getImage() const noexcept { return image_; }
    [[nodiscard]] const std::vector<cv::KeyPoint> &getKeypoints() const noexcept { return keypoints_; }
    [[nodiscard]] const cv::Mat &getDescriptors() const noexcept { return descriptors_; }
    [[nodiscard]] const ViewPoint<T> &getViewPoint() const { return viewpoint_.value(); }
    [[nodiscard]] T getScore() const { return viewpoint_.value().getScore(); }

    // Setters
    void setImage(cv::Mat image) { image_ = validateImage(std::move(image)); }
    void setKeypoints(std::vector<cv::KeyPoint> keypoints) noexcept { keypoints_ = std::move(keypoints); }
    void setDescriptors(cv::Mat descriptors) noexcept { descriptors_ = std::move(descriptors); }
    void setViewPoint(ViewPoint<T> viewpoint) noexcept { viewpoint_ = std::move(viewpoint); }
    void setScore(T score) { viewpoint_.value().setScore(score); }

    // Helper methods
    [[nodiscard]] bool hasViewPoint() const noexcept { return viewpoint_.has_value(); }

    // Serialization to string
    [[nodiscard]] std::string toString() const {
        return fmt::format("Image(size: {}x{}, keypoints: {}, descriptors: {}x{})", image_.cols, image_.rows,
                           keypoints_.size(), descriptors_.rows, descriptors_.cols);
    }

private:
    cv::Mat image_, descriptors_;
    std::vector<cv::KeyPoint> keypoints_;
    std::optional<ViewPoint<T>> viewpoint_;

    static cv::Mat validateImage(cv::Mat image) {
        if (image.empty()) {
            LOG_ERROR("Image data is empty. Cannot set image.");
            throw std::invalid_argument("Image data cannot be empty.");
        }
        return image;
    }

    void detect(const std::unique_ptr<FeatureExtractor> &extractor) {
        auto [keypoints, descriptors] = extractor->extract(image_);
        keypoints_ = std::move(keypoints);
        descriptors_ = std::move(descriptors);
    }

    // Feature detection
    void detect(const cv::Ptr<cv::Feature2D> &detector) {
        detector->detectAndCompute(image_, cv::noArray(), keypoints_, descriptors_);
    }
};

#endif // TYPE_IMAGE_HPP
