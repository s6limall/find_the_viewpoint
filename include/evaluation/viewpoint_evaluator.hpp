// File: evaluation/viewpoint_evaluator.hpp

#ifndef VIEWPOINT_EVALUATOR_HPP
#define VIEWPOINT_EVALUATOR_HPP

#include <vector>
#include <opencv2/opencv.hpp>

#include "evaluation/evaluator.hpp"
#include "core/perception.hpp"
#include "common/logging/logger.hpp"
#include "config/configuration.hpp"

namespace evaluation {
    class ViewpointEvaluator final : public Evaluator {
    public:
        [[nodiscard]] std::vector<double> evaluate(const std::vector<std::vector<double> > &samples,
                                                   const cv::Mat &target_image) const override;

    private:
        [[nodiscard]] static double computeSSIM(const cv::Mat &image1, const cv::Mat &image2);
    };
}

#endif //VIEWPOINT_EVALUATOR_HPP
