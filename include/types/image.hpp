#ifndef TYPE_IMAGE_HPP
#define TYPE_IMAGE_HPP

#include <concepts>
#include <fmt/core.h>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <stdexcept>

#include "common/logging/logger.hpp"
#include "config/configuration.hpp"
#include "core/eye.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/extractor/akaze_extractor.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image_preprocessor.hpp"
#include "types/concepts.hpp"
#include "types/viewpoint.hpp"


template<Arithmetic T = double>
class Image {
public:
    struct Features {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    Image() = default;

    explicit Image(const cv::Mat &image,
                   std::shared_ptr<processing::image::FeatureExtractor> extractor = nullptr) noexcept :
        extractor_(std::move(extractor)) {
        setImage(image);
    }

    // Rule of five
    Image(const Image &) = delete;
    Image &operator=(const Image &other) {
        if (this != &other) {
            image_ = other.image_.clone();
            features_ = other.features_;
            viewpoint_ = other.viewpoint_;
            extractor_ = other.extractor_;
        }
        return *this;
    }
    Image(Image &&) noexcept = default;
    Image &operator=(Image &&other) noexcept {
        if (this != &other) {
            image_ = std::move(other.image_);
            features_ = std::move(other.features_);
            viewpoint_ = std::move(other.viewpoint_);
            extractor_ = std::move(other.extractor_);
        }
        return *this;
    }
    ~Image() = default;

    [[nodiscard]] const cv::Mat &getImage() const noexcept { return image_; }
    [[nodiscard]] const std::vector<cv::KeyPoint> &getKeypoints() const { return ensureFeatures().keypoints; }
    [[nodiscard]] const cv::Mat &getDescriptors() const { return ensureFeatures().descriptors; }

    [[nodiscard]] const std::optional<ViewPoint<T>> &getViewPoint() const noexcept { return viewpoint_; }
    [[nodiscard]] T getScore() const noexcept { return viewpoint_.has_value() ? viewpoint_->getScore() : T{}; }
    [[nodiscard]] T getUncertainty() const noexcept {
        return viewpoint_.has_value() ? viewpoint_->getUncertainty() : T{};
    }

    void setImage(const cv::Mat &image) noexcept {
        if (config::get("image.preprocess", false)) {
            static ImagePreprocessor preprocessor;
            image_ = preprocessor.process(image);
        } else {
            image_ = image.clone();
        }
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

    [[nodiscard]] constexpr bool hasViewPoint() const noexcept { return viewpoint_.has_value(); }
    [[nodiscard]] constexpr bool hasFeatures() const noexcept { return features_.has_value(); }

    [[nodiscard]] static Image<T>
    fromViewPoint(ViewPoint<T> viewpoint, std::shared_ptr<processing::image::FeatureExtractor> extractor = nullptr) {
        cv::Mat rendered_image = core::Eye::render(viewpoint.toView().getPose());
        Image<T> image(std::move(rendered_image), std::move(extractor));
        image.setViewPoint(std::move(viewpoint));
        return image;
    }

    [[nodiscard]] std::string toString() const {
        return fmt::format("Image(size: {}x{}, features: {}, viewpoint: {})", image_.cols, image_.rows,
                           features_.has_value()
                                   ? fmt::format("keypoints: {}, descriptors: {}x{}", features_->keypoints.size(),
                                                 features_->descriptors.rows, features_->descriptors.cols)
                                   : "not computed",
                           viewpoint_.has_value() ? viewpoint_->toString() : "not set");
    }

    bool operator==(const Image &other) const {
        return image_.size() == other.image_.size() && cv::countNonZero(image_ != other.image_) == 0 &&
               viewpoint_ == other.viewpoint_;
    }

    bool operator!=(const Image &other) const { return !(*this == other); }

private:
    cv::Mat image_;
    mutable std::optional<Features> features_;
    std::optional<ViewPoint<T>> viewpoint_;
    mutable std::shared_ptr<processing::image::FeatureExtractor> extractor_;

    const Features &ensureFeatures() const {
        if (!features_) {
            if (!extractor_) {
                LOG_WARN("Feature extractor not set. Attempting to read configuration.");

                const auto feature_extractor = config::get("image.feature.extractor.type", "SIFT");
                if (feature_extractor == "SIFT") {
                    extractor_ = std::make_shared<processing::image::SIFTExtractor>();
                } else if (feature_extractor == "ORB") {
                    extractor_ = std::make_shared<processing::image::ORBExtractor>();
                } else if (feature_extractor == "AKAZE") {
                    extractor_ = std::make_shared<processing::image::AKAZEExtractor>();
                } else {
                    throw std::runtime_error("Invalid feature extractor type: " + feature_extractor);
                }
            }

            auto [keypoints, descriptors] = extractor_->extract(image_);
            features_.emplace(Features{std::move(keypoints), std::move(descriptors)});
        }
        return features_.value();
    }
};

#endif // TYPE_IMAGE_HPP
