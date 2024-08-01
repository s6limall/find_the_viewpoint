// File: processing/image/comparator/feature_comparator.hpp

#ifndef FEATURE_COMPARATOR_HPP
#define FEATURE_COMPARATOR_HPP

#include "processing/image/comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"
#include "processing/image/ransac.hpp"

namespace processing::image {

    class FeatureComparator final : public ImageComparator {
    public:
        FeatureComparator(std::shared_ptr<FeatureExtractor> extractor, std::shared_ptr<FeatureMatcher> matcher);

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override;
        [[nodiscard]] double compare(const Image<> &image1, const Image<> &image2) const override;

    private:
        std::shared_ptr<FeatureExtractor> extractor_;
        std::shared_ptr<FeatureMatcher> matcher_;

        using Point2D = Eigen::Vector2d;
        using Homography = Eigen::Matrix3d;

        double computeSimilarity(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const;
        Homography findHomographyRANSAC(const std::vector<Point2D> &points1, const std::vector<Point2D> &points2) const;
    };


} // namespace processing::image

#endif // FEATURE_COMPARATOR_HPP
