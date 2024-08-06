// File: types/image.hpp

#ifndef TYPE_IMAGE_HPP
#define TYPE_IMAGE_HPP

#include <concepts>
#include <fmt/core.h>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/logging/logger.hpp"
#include "common/utilities/image.hpp"
#include "core/perception.hpp"
#include "processing/image/feature/extractor.hpp"
#include "types/viewpoint.hpp"


template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T = double>
class Image {
public:
    struct Features {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    Image() = default;

    explicit Image(cv::Mat image, std::shared_ptr<processing::image::FeatureExtractor> extractor = nullptr) :
        image_{std::move(image)}, extractor_{std::move(extractor)} {
        if (image_.empty()) {
            throw std::invalid_argument("Image data cannot be empty.");
        }
    }

    // Rule of five
    Image(const Image &) = delete;
    Image &operator=(const Image &) = delete;
    Image(Image &&) noexcept = default;
    Image &operator=(Image &&) noexcept = default;
    ~Image() = default;

    // Getters
    [[nodiscard]] const cv::Mat &getImage() const noexcept { return image_; }
    [[nodiscard]] const std::vector<cv::KeyPoint> getKeypoints() const { return ensureFeatures().keypoints; }
    [[nodiscard]] const cv::Mat &getDescriptors() const { return ensureFeatures().descriptors; }
    [[nodiscard]] const cv::Mat &getHash() const {
        if (!hash_) {
            hash_ = common::utilities::computePerceptualHash(image_);
        }
        return *hash_;
    }

    [[nodiscard]] const std::optional<ViewPoint<T>> &getViewPoint() const noexcept { return viewpoint_; }
    [[nodiscard]] T getScore() const noexcept { return viewpoint_ ? viewpoint_->getScore() : T{}; }
    [[nodiscard]] T getUncertainty() const noexcept { return viewpoint_ ? viewpoint_->getUncertainty() : T{}; }

    // Setters
    void setImage(cv::Mat image) {
        if (image.empty()) {
            throw std::invalid_argument("Image data cannot be empty.");
        }
        image_ = std::move(image);
        hash_.reset();
        features_.reset();
    }

    void setExtractor(std::shared_ptr<processing::image::FeatureExtractor> extractor) noexcept {
        extractor_ = std::move(extractor);
        features_.reset();
    }

    void setViewPoint(ViewPoint<T> viewpoint) noexcept { viewpoint_ = std::move(viewpoint); }
    void setScore(T score) noexcept {
        if (viewpoint_)
            viewpoint_->setScore(score);
    }
    void setUncertainty(T uncertainty) noexcept {
        if (viewpoint_)
            viewpoint_->setUncertainty(uncertainty);
    }

    [[nodiscard]] bool hasViewPoint() const noexcept { return viewpoint_.has_value(); }
    [[nodiscard]] bool hasFeatures() const noexcept { return features_.has_value(); }

    // Static factory method
    [[nodiscard]] static Image<T>
    fromViewPoint(ViewPoint<T> viewpoint, std::shared_ptr<processing::image::FeatureExtractor> extractor = nullptr) {
        cv::Mat rendered_image = core::Perception::render(viewpoint.toView().getPose());
        Image<T> image(std::move(rendered_image), std::move(extractor));
        image.setViewPoint(std::move(viewpoint));
        return image;
    }

    // Serialization to string
    [[nodiscard]] std::string toString() const {
        return fmt::format("Image(size: {}x{}, features: {})", image_.cols, image_.rows,
                           features_ ? fmt::format("keypoints: {}, descriptors: {}x{}", features_->keypoints.size(),
                                                   features_->descriptors.rows, features_->descriptors.cols)
                                     : "not computed");
    }

private:
    cv::Mat image_;
    mutable std::optional<Features> features_;
    mutable std::optional<cv::Mat> hash_;
    std::optional<ViewPoint<T>> viewpoint_;
    std::shared_ptr<processing::image::FeatureExtractor> extractor_;

    const Features &ensureFeatures() const {
        if (!features_) {
            if (!extractor_) {
                LOG_ERROR("Feature extractor not set");
                throw std::runtime_error("Feature extractor not set");
            }
            LOG_INFO("Computing features using custom extractor.");
            auto [keypoints, descriptors] = extractor_->extract(image_);
            features_ = Features{std::move(keypoints), std::move(descriptors)};
        }
        return *features_;
    }
};


#endif // TYPE_IMAGE_HPP
