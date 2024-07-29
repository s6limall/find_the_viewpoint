// File: types/image.hpp

#ifndef TYPE_IMAGE_HPP
#define TYPE_IMAGE_HPP

#include <fmt/core.h>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/logging/logger.hpp"
#include "common/utilities/image.hpp"
#include "processing/image/feature/extractor.hpp"
#include "types/viewpoint.hpp"

using FeatureExtractor = processing::image::FeatureExtractor;

template<typename T = double>
class Image {
public:
    Image() noexcept = default;

    Image(cv::Mat image, const std::shared_ptr<FeatureExtractor> &extractor) :
        image_{validateImage(std::move(image))}, hash_once_flag_(std::make_shared<std::once_flag>()) {
        detect(extractor);
    }

    explicit Image(cv::Mat image, const cv::Ptr<cv::Feature2D> &detector = cv::ORB::create()) :
        image_{validateImage(std::move(image))}, hash_once_flag_(std::make_shared<std::once_flag>()) {
        detect(detector);
    }

    Image(cv::Mat image, cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints) :
        image_{validateImage(std::move(image))}, descriptors_{std::move(descriptors)}, keypoints_{std::move(keypoints)},
        hash_once_flag_(std::make_shared<std::once_flag>()) {}

    // Copy constructor
    Image(const Image &other) :
        image_(other.image_), descriptors_(other.descriptors_), keypoints_(other.keypoints_), hash_(other.hash_),
        viewpoint_(other.viewpoint_), hash_once_flag_(std::make_shared<std::once_flag>()) {}

    // Copy assignment operator
    Image &operator=(const Image &other) {
        if (this != &other) {
            image_ = other.image_;
            descriptors_ = other.descriptors_;
            keypoints_ = other.keypoints_;
            hash_ = other.hash_;
            viewpoint_ = other.viewpoint_;
            hash_once_flag_ = std::make_shared<std::once_flag>();
        }
        return *this;
    }

    // Move constructor and assignment operator
    Image(Image &&other) noexcept = default;
    Image &operator=(Image &&other) noexcept = default;

    // Getters
    [[nodiscard]] const cv::Mat &getImage() const noexcept { return image_; }
    [[nodiscard]] const std::vector<cv::KeyPoint> &getKeypoints() const noexcept { return keypoints_; }
    [[nodiscard]] const cv::Mat &getDescriptors() const noexcept { return descriptors_; }
    [[nodiscard]] const cv::Mat &getHash() const noexcept {
        std::call_once(hash_once_flag_, &Image::computeHash, this);
        return *hash_;
    }

    [[nodiscard]] const ViewPoint<T> &getViewPoint() const { return viewpoint_.value(); }
    [[nodiscard]] T getScore() const noexcept { return viewpoint_ ? viewpoint_->getScore() : T(); }
    [[nodiscard]] T getUncertainty() const noexcept { return viewpoint_ ? viewpoint_->getUncertainty() : T(); }

    // Setters
    void setImage(cv::Mat image) {
        image_ = validateImage(std::move(image));
        hash_.reset();
        hash_once_flag_ = std::make_shared<std::once_flag>();
    }

    void setKeypoints(std::vector<cv::KeyPoint> keypoints) noexcept { keypoints_ = std::move(keypoints); }
    void setDescriptors(cv::Mat descriptors) noexcept { descriptors_ = std::move(descriptors); }
    void setViewPoint(ViewPoint<T> viewpoint) noexcept { viewpoint_ = std::move(viewpoint); }
    void setScore(T score) noexcept {
        if (viewpoint_)
            viewpoint_->setScore(score);
    }
    void setUncertainty(T uncertainty) noexcept {
        if (viewpoint_)
            viewpoint_->setUncertainty(uncertainty);
    }

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

    /*
     * mutable: allows the hash to be computed & cached on first request,
     * even if the getHash() is called on a const object
     */
    mutable std::optional<cv::Mat> hash_;
    mutable std::shared_ptr<std::once_flag> hash_once_flag_;
    std::optional<ViewPoint<T>> viewpoint_;

    static cv::Mat validateImage(cv::Mat image) {
        if (image.empty()) {
            LOG_ERROR("Image data is empty. Cannot set image.");
            throw std::invalid_argument("Image data cannot be empty.");
        }
        return image;
    }

    void detect(const std::shared_ptr<FeatureExtractor> &extractor) {
        auto [keypoints, descriptors] = extractor->extract(image_);
        keypoints_ = std::move(keypoints);
        descriptors_ = std::move(descriptors);
    }

    void detect(const cv::Ptr<cv::Feature2D> &detector) {
        detector->detectAndCompute(image_, cv::noArray(), keypoints_, descriptors_);
    }

    void computeHash() const { hash_ = common::utilities::computePerceptualHash(image_); }
};

#endif // TYPE_IMAGE_HPP
