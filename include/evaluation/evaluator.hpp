// File: evaluation/evaluator.hpp

#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include <vector>
#include <opencv2/opencv.hpp>

namespace evaluation {

    class Evaluator {
    public:
        virtual ~Evaluator() = default;

        [[nodiscard]] virtual std::vector<double> evaluate(const std::vector<std::vector<double> > &samples,
                                                           const cv::Mat &target_image) const = 0;
    };

}


#endif //EVALUATOR_HPP
