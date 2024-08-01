#include "processing/image/comparison/feature_comparator.hpp"

namespace processing::image {

    FeatureComparator::FeatureComparator(std::shared_ptr<FeatureExtractor> extractor,
                                         std::shared_ptr<FeatureMatcher> matcher) :
        extractor_(std::move(extractor)), matcher_(std::move(matcher)) {}

    double FeatureComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        try {
            auto [keypoints1, descriptors1] = extractor_->extract(image1);
            auto [keypoints2, descriptors2] = extractor_->extract(image2);
            return computeSimilarity(descriptors1, descriptors2);
        } catch (const std::exception &e) {
            LOG_ERROR("Error comparing images: {}", e.what());
            return error_score_;
        }
    }

    double FeatureComparator::compare(const Image<> &image1, const Image<> &image2) const {
        try {
            const auto &descriptors1 = image1.getDescriptors();
            const auto &descriptors2 = image2.getDescriptors();
            return computeSimilarity(descriptors1, descriptors2);
        } catch (const std::exception &e) {
            LOG_ERROR("Error comparing images: {}", e.what());
            return error_score_;
        }
    }

    double FeatureComparator::computeSimilarity(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const {
        if (descriptors1.empty() || descriptors2.empty()) {
            LOG_ERROR("One or both sets of descriptors are empty.");
            return error_score_;
        }

        const auto matches = matcher_->match(descriptors1, descriptors2);

        if (matches.empty()) {
            LOG_WARN("No matches found.");
            return 0.0;
        }


        // TODO: REMOVE
        // Calculate a score based on the matches without performing RANSAC
        /*double total_distance = 0.0;
        for (const auto &match: matches) {
            total_distance += match.distance;
        }
        double average_distance = total_distance / matches.size();

        // Return a score inversely proportional to the average distance
        double similarity_score = 1.0 / (1.0 + average_distance);

        return similarity_score;*/
        // ---


        // Extract matching points
        std::vector<Point2D> points1, points2;
        points1.reserve(matches.size());
        points2.reserve(matches.size());

        for (const auto &match: matches) {
            points1.emplace_back(descriptors1.row(match.queryIdx).at<float>(0),
                                 descriptors1.row(match.queryIdx).at<float>(1));
            points2.emplace_back(descriptors2.row(match.trainIdx).at<float>(0),
                                 descriptors2.row(match.trainIdx).at<float>(1));
        }

        // Apply RANSAC to find the robust homography
        auto homography = findHomographyRANSAC(points1, points2);
        if (homography.isZero()) {
            return 0.0;
        }

        double total_distance = 0.0;
        for (const auto &match: matches) {
            Point2D pt1(descriptors1.row(match.queryIdx).at<float>(0), descriptors1.row(match.queryIdx).at<float>(1));
            Point2D pt2(descriptors2.row(match.trainIdx).at<float>(0), descriptors2.row(match.trainIdx).at<float>(1));
            Eigen::Vector3d pt1_homogeneous = pt1.homogeneous();
            Eigen::Vector3d pt2_transformed_homogeneous = homography * pt1_homogeneous;
            Point2D pt2_transformed = pt2_transformed_homogeneous.hnormalized();
            total_distance += (pt2 - pt2_transformed).squaredNorm();
        }

        return 1.0 / (1.0 + total_distance / matches.size()); // Normalized similarity score between 0 and 1
    }

    FeatureComparator::Homography FeatureComparator::findHomographyRANSAC(const std::vector<Point2D> &points1,
                                                                          const std::vector<Point2D> &points2) const {
        using ModelType = Homography;

        auto model_fitter = [](const std::vector<Point2D> &points) -> ModelType {
            Eigen::MatrixXd A(2 * points.size(), 9);
            for (size_t i = 0; i < points.size(); ++i) {
                const auto &pt1 = points[i];
                const auto &pt2 = points[i];
                A.row(2 * i) << pt1.x(), pt1.y(), 1, 0, 0, 0, -pt2.x() * pt1.x(), -pt2.x() * pt1.y(), -pt2.x();
                A.row(2 * i + 1) << 0, 0, 0, pt1.x(), pt1.y(), 1, -pt2.y() * pt1.x(), -pt2.y() * pt1.y(), -pt2.y();
            }
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
            Eigen::VectorXd h = svd.matrixV().col(8);
            return Eigen::Map<Eigen::Matrix3d>(h.data());
        };

        auto distance_calculator = [](const Point2D &point, const ModelType &model) -> double {
            Eigen::Vector3d pt1_homogeneous = point.homogeneous();
            Eigen::Vector3d pt1_transformed_homogeneous = model * pt1_homogeneous;
            Point2D pt1_transformed = pt1_transformed_homogeneous.hnormalized();
            return (point - pt1_transformed).squaredNorm();
        };

        RANSAC<Point2D, ModelType> ransac(model_fitter, distance_calculator);
        auto result = ransac.findDominantPlane(points1);

        return result ? result->model : ModelType::Zero();
    }

} // namespace processing::image
