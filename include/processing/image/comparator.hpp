// File: processing/image/comparator.hpp

#ifndef IMAGE_COMPARATOR_HPP
#define IMAGE_COMPARATOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "comparison/peak_snr_comparator.hpp"
#include "feature/matcher.hpp"
#include "types/image.hpp"

namespace processing::image {

    class ImageComparator {
    public:
        virtual ~ImageComparator() = default;

        // Compare two images and return a score indicating their similarity. [0, 1] where 0 means no similarity.
        [[nodiscard]] virtual double compare(const cv::Mat &image1, const cv::Mat &image2) const = 0;
        [[nodiscard]] virtual double compare(const Image<> &image1, const Image<> &image2) const = 0;

        // Returns comparator and target score
        static std::pair<std::shared_ptr<ImageComparator>, double>
        create(const std::shared_ptr<FeatureExtractor> &extractor, const std::shared_ptr<FeatureMatcher> &matcher);


    protected:
        // Default maximum value to indicate errors.
        static constexpr double error_score_ = std::numeric_limits<double>::max();
        static constexpr double epsilon_ = std::numeric_limits<double>::epsilon();

        // Normalize the score using the PSNR comparator
        [[nodiscard]] static double normalize(const cv::Mat &image1, const cv::Mat &image2, const double score,
                                              const double max_score = 1.0) {

            if (!config::get("image.comparator.normalize", false)) {
                return score;
            }

            LOG_DEBUG("Normalizing score using PSNR.");
            const double psnr = PeakSNRComparator::compare(image1, image2);

            if (psnr > 0) {
                // scale the score relative to its maximum possible value
                double normalized_score = score / max_score;

                // scale based on the PSNR value, ensuring it doesn't exceed 1
                normalized_score *= psnr;
                LOG_INFO("Score (normalized): {}, Score (original): {}, PSNR: {}", normalized_score, score, psnr);

                return std::min(normalized_score, 1.0);
            }

            return 0.0; // In case PSNR is 0 (SHOULD NOT HAPPEN)
        }
    };

} // namespace processing::image

#endif // IMAGE_COMPARATOR_HPP
