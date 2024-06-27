// File: processing/image/feature/extractor/sift.cpp

#include "processing/image/feature/extractor/sift_extractor.hpp"

#include "config/configuration.hpp"

namespace processing::image {
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> SIFTExtractor::extract(const cv::Mat &image) const {
        if (image.empty()) {
            throw std::invalid_argument("Input image is empty");
        }

        const auto &config = config::Configuration::getInstance();
        int n_features = config.get<int>("feature_detector.sift.n_features", 0);
        int n_octave_layers = config.get<int>("feature_detector.sift.n_octave_layers", 3);
        double contrast_threshold = config.get<double>("feature_detector.sift.contrast_threshold", 0.04);
        double edge_threshold = config.get<double>("feature_detector.sift.edge_threshold", 10);
        double sigma = config.get<double>("feature_detector.sift.sigma", 1.6);

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(
            n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma
        );

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        return {keypoints, descriptors};
    }
}
