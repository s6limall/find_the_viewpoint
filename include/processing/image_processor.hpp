// File: processing/image_processor.hpp


#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <type_traits>
#include <variant>
#include <vector>
#include "common/logging/logger.hpp"
#include "common/utilities/image.hpp"
#include "types/image.hpp"

namespace processing::image {

    using ImageVariant = std::variant<cv::Mat, Image<>>;

    template<typename... Ts>
    struct always_false : std::false_type {};

    class ImageProcessor {
    public:
        ImageProcessor() = default;

        // Compare two images by perceptual hash
        template<typename ImageType1, typename ImageType2>
        [[nodiscard]] static std::pair<bool, double> compareImagesByHash(const ImageType1 &image1,
                                                                         const ImageType2 &image2,
                                                                         double similarity_threshold = 0.9) noexcept;

        // Compute similarity scores for a target image against multiple candidate images
        template<typename ImageType>
        [[nodiscard]] static std::vector<double>
        computeSimilarityScores(const ImageType &targetImage, const std::vector<ImageVariant> &candidate_images,
                                double similarity_threshold = 0.9) noexcept;

    private:
        template<typename ImageType>
        [[nodiscard]] static cv::Mat extractImage(const ImageType &image) noexcept;

        [[nodiscard]] static double computeHammingDistance(const cv::Mat &hash1, const cv::Mat &hash2) noexcept;

        template<typename ImageType>
        [[nodiscard]] static cv::Mat getHash(const ImageType &image) noexcept;
    };

    // Implementation

    template<typename ImageType>
    cv::Mat ImageProcessor::extractImage(const ImageType &image) noexcept {
        LOG_DEBUG("Extracting image");
        if constexpr (std::is_same_v<ImageType, cv::Mat>) {
            LOG_INFO("Image is already in OpenCV format (cv::Mat)");
            return image;
        } else if constexpr (std::is_same_v<ImageType, Image<>>) {
            LOG_INFO("Image is in custom format (Image<>), converting to OpenCV format (cv::Mat)");
            return image.getImage();
        } else {
            LOG_ERROR("Unsupported image type provided");
            static_assert(always_false<ImageType>::value, "Unsupported image type.");
            return {}; // Added to ensure all paths return a value
        }
    }

    inline double ImageProcessor::computeHammingDistance(const cv::Mat &hash1, const cv::Mat &hash2) noexcept {
        return static_cast<double>(cv::norm(hash1, hash2, cv::NORM_HAMMING));
    }

    template<typename ImageType>
    cv::Mat ImageProcessor::getHash(const ImageType &image) noexcept {
        if constexpr (std::is_same_v<ImageType, Image<>>) {
            LOG_TRACE("Image is in custom format (Image<>), using perceptual hash from Image<> object.");
            return image.getHash();
        } else {
            LOG_TRACE("Image is in OpenCV format (cv::Mat), computing perceptual hash.");
            return common::utilities::computePerceptualHash(extractImage(image));
        }
    }

    template<typename ImageType1, typename ImageType2>
    std::pair<bool, double> ImageProcessor::compareImagesByHash(const ImageType1 &image1, const ImageType2 &image2,
                                                                const double similarity_threshold) noexcept {
        LOG_DEBUG("Retrieving/computing perceptual hash for images");
        const cv::Mat hash1 = getHash(image1), hash2 = getHash(image2);

        const double distance = computeHammingDistance(hash1, hash2);
        LOG_DEBUG("Hamming distance: {}", distance);

        const double similarity = 1.0 - distance / static_cast<double>(hash1.total() * 8); // Hash size in bits
        const bool match = similarity >= similarity_threshold;

        LOG_DEBUG("Images are {}similar (similarity: {}, similarity threshold: {})", match ? "" : "not ", similarity,
                  similarity_threshold);
        return {match, similarity};
    }

    template<typename ImageType>
    std::vector<double> ImageProcessor::computeSimilarityScores(const ImageType &targetImage,
                                                                const std::vector<ImageVariant> &candidate_images,
                                                                double similarity_threshold) noexcept {
        std::vector<double> scores;
        scores.reserve(candidate_images.size());

        LOG_DEBUG("Computing similarity scores for target image");
        for (const auto &candidate: candidate_images) {
            auto [_, similarity] = std::visit(
                    [&](const auto &img) { return compareImagesByHash(targetImage, img, similarity_threshold); },
                    candidate);
            scores.push_back(similarity);
        }

        return scores;
    }

} // namespace processing::image

#endif // IMAGE_PROCESSOR_HPP
