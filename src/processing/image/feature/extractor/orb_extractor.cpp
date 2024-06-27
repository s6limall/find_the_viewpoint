// File: processing/image/feature/extractor/orb.cpp

#include "processing/image/feature/extractor/orb_extractor.hpp"

#include "config/configuration.hpp"

namespace processing::image {
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> ORBExtractor::extract(const cv::Mat &image) const {
        if (image.empty()) {
            throw std::invalid_argument("Input image is empty");
        }

        const auto &config = config::Configuration::getInstance();
        int n_features = config.get<int>("feature_detector.orb.n_features", 500);
        float scale_factor = config.get<float>("feature_detector.orb.scale_factor", 1.2f);
        int n_levels = config.get<int>("feature_detector.orb.n_levels", 8);
        int edge_threshold = config.get<int>("feature_detector.orb.edge_threshold", 31);
        int first_level = config.get<int>("feature_detector.orb.first_level", 0);
        int wta_k = config.get<int>("feature_detector.orb.wta_k", 2);
        cv::ORB::ScoreType score_type = static_cast<cv::ORB::ScoreType>(
            config.get<int>("feature_detector.orb.score_type", cv::ORB::HARRIS_SCORE)
        );
        int patch_size = config.get<int>("feature_detector.orb.patch_size", 31);
        int fast_threshold = config.get<int>("feature_detector.orb.fast_threshold", 20);

        cv::Ptr<cv::ORB> orb = cv::ORB::create(
            n_features, scale_factor, n_levels, edge_threshold, first_level,
            wta_k, score_type, patch_size, fast_threshold
        );

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        return {keypoints, descriptors};
    }
}
