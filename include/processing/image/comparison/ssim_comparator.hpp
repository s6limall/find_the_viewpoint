// File: processing/image/comparison/ssim_comparator.hpp

#ifndef IMAGE_COMPARATOR_SSIM_HPP
#define IMAGE_COMPARATOR_SSIM_HPP

#include <numeric>
#include <opencv2/quality/qualityssim.hpp>
#include "processing/image/comparator.hpp"

namespace processing::image {

    class SSIMComparator final : public ImageComparator {
    public:
        SSIMComparator() = default;

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override {
            if (!isValidInput(image1, image2)) {
                LOG_WARN("Images are empty, or their sizes/types do not match.");
                return error_score_;
            }
            return computeSSIM(image1, image2);
        }

        [[nodiscard]] double compare(const Image<> &img1, const Image<> &img2) const override {
            return compare(img1.getImage(), img2.getImage());
        }

    private:
        [[nodiscard]] static inline bool isValidInput(const cv::Mat &image1, const cv::Mat &image2) noexcept {
            return !image1.empty() && !image2.empty() && (image1.size() == image2.size()) &&
                   (image1.type() == image2.type());
        }

        [[nodiscard]] static double computeSSIM(const cv::Mat &img1, const cv::Mat &img2) noexcept {
            if (img1.channels() == 1) {
                return cv::quality::QualitySSIM::compute(img1, img2, cv::noArray())[0];
            }

            // Convert images to 32-bit float for accurate SSIM computation
            cv::Mat img1_f, img2_f;
            img1.convertTo(img1_f, CV_32F);
            img2.convertTo(img2_f, CV_32F);

            // OpenCV's SSIM computation returns a cv::Scalar for multi-channel images
            const cv::Scalar ssim_result = cv::quality::QualitySSIM::compute(img1_f, img2_f, cv::noArray());

            // Efficiently compute the mean SSIM across channels using std::transform_reduce
            return std::reduce(ssim_result.val, ssim_result.val + img1.channels(), 0.0) / img1.channels();
        }
    };

} // namespace processing::image

#endif // IMAGE_COMPARATOR_SSIM_HPP
