#include "evaluation/viewpoint_evaluator.hpp"

#include <opencv2/quality/qualityssim.hpp>

namespace evaluation {

    std::vector<double> ViewpointEvaluator::evaluate(const std::vector<std::vector<double> > &samples,
                                                     const cv::Mat &target_image) const {
        std::vector<double> ssim_scores;
        ssim_scores.reserve(samples.size()); // Reserve space for efficiency

        for (const auto &sample: samples) {
            Eigen::Vector3d position(sample[0], sample[1], sample[2]);
            core::View view;
            view.computePose(position, {0.0, 0.0, 0.0});
            cv::Mat rendered_image = core::Perception::render(view.getPose());

            if (rendered_image.empty()) {
                LOG_ERROR("Failed to capture rendered image.");
                continue;
            }
            // cv::cvtColor(rendered_image, rendered_image, cv::COLOR_BGR2GRAY);

            double ssim_score = computeSSIM(target_image, rendered_image);
            ssim_scores.push_back(ssim_score);
            LOG_DEBUG("Computed SSIM score {} for sample at position ({}, {}, {})", ssim_score, sample[0], sample[1],
                      sample[2]);
        }

        return ssim_scores;
    }

    double ViewpointEvaluator::computeSSIM(const cv::Mat &image1, const cv::Mat &image2) {
        cv::Mat ssim_map;
        cv::Scalar ssim_score = cv::quality::QualitySSIM::compute(image1, image2, ssim_map);
        return ssim_score[0];
    }

}
