// File: include/viewpoint/evaluator.hpp

#ifndef VIEWPOINT_EVALUATOR_HPP
#define VIEWPOINT_EVALUATOR_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"
#include "processing/image/comparison.hpp"

namespace viewpoint {

    class Evaluator {
    public:
        Evaluator();

        // Evaluates a sample viewpoint and returns a fitness score.
        double evaluate(const std::vector<double>& sample, const cv::Mat& target_image, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) const;

        // Refines the search space based on evaluated samples.
        std::vector<double> refineSearchSpace(
            const std::vector<std::vector<double>>& samples,
            const std::vector<double>& lower_bounds,
            const std::vector<double>& upper_bounds) const;

        // Evaluates a set of samples and provides feedback for refinement.
        void evaluateAndRefine(
            const std::vector<std::vector<double>>& samples,
            std::vector<double>& lower_bounds,
            std::vector<double>& upper_bounds,
            const cv::Mat& target_image,
            const cv::Mat& camera_matrix,
            const cv::Mat& dist_coeffs) const;

    private:
        // Use existing feature extractors and matchers.
        std::unique_ptr<processing::image::FeatureExtractor> extractor_;
        std::unique_ptr<processing::image::FeatureMatcher> matcher_;

        // Calculate the center of mass of samples to refine the search space.
        std::vector<double> calculateCenterOfMass(const std::vector<std::vector<double>>& samples) const;

        // Adjusts the bounds based on the feedback from the evaluation.
        void adjustBounds(
            std::vector<double>& lower_bounds,
            std::vector<double>& upper_bounds,
            const std::vector<double>& center) const;

        // Helper function to compute similarity using SSIM.
        double computeSSIM(const cv::Mat& image1, const cv::Mat& image2) const;

        // Initialize the feature extractor and matcher.
        void initializeFeatureTools();
    };

}

#endif // VIEWPOINT_EVALUATOR_HPP
