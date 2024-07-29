// File: include/viewpoint/evaluator.hpp

#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include <vector>
#include "core/perception.hpp"
#include "processing/image/comparator.hpp"
#include "types/image.hpp"

namespace viewpoint {
    template<typename T = double>
    class Evaluator {
    public:
        Evaluator(const Image<T> &target_image, std::shared_ptr<processing::image::ImageComparator> comparator);

        double evaluate(const Image<T> &image) const;

    private:
        Image<T> target_image_;
        std::shared_ptr<processing::image::ImageComparator> comparator_;
    };

    // Definitions

    template<typename T>
    Evaluator<T>::Evaluator(const Image<T> &target_image,
                            std::shared_ptr<processing::image::ImageComparator> comparator) :
        target_image_(target_image), comparator_(std::move(comparator)) {}

    template<typename T>
    double Evaluator<T>::evaluate(const Image<T> &image) const {
        return comparator_->compare(target_image_, image);
    }
} // namespace viewpoint

#endif // EVALUATOR_HPP
