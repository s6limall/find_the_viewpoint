// File: include/viewpoint/evaluator.hpp

#ifndef VIEWPOINT_EVALUATOR_HPP
#define VIEWPOINT_EVALUATOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "core/perception.hpp"
#include "processing/image/comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"
#include "types/image.hpp"

namespace viewpoint {

    template<typename T = double>
    class Evaluator {
    public:
        explicit Evaluator(Image<T> target_image) : target_image_(target_image) {}

        std::vector<Image<T>> evaluate(const std::unique_ptr<processing::image::ImageComparator> &comparator,
                                       const std::vector<ViewPoint<T>> &samples);

    private:
        Image<T> target_image_;
    };

    template<typename T>
    std::vector<Image<T>> Evaluator<T>::evaluate(const std::unique_ptr<processing::image::ImageComparator> &comparator,
                                                 const std::vector<ViewPoint<T>> &samples) {

        std::vector<Image<>> evaluated_images;
        evaluated_images.reserve(samples.size());

        for (const auto &sample: samples) {
            core::View view = sample.toView();
            cv::Mat rendered_image = core::Perception::render(view.getPose());

            // Create Image object for the rendered image
            Image<> rendered_sample_image(rendered_image, cv::ORB::create());
            rendered_sample_image.setViewPoint(sample);

            // Compute score
            const double score = comparator->compare(target_image_.getImage(), rendered_image);
            rendered_sample_image.setScore(score);

            evaluated_images.push_back(rendered_sample_image);
        }

        return evaluated_images;
    }

} // namespace viewpoint

#endif // VIEWPOINT_EVALUATOR_HPP
