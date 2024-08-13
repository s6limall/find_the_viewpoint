// File: optimization/radius_refiner.hpp

#ifndef RADIUS_REFINER_HPP
#define RADIUS_REFINER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <opencv2/opencv.hpp>
#include "common/logging/logger.hpp"
#include "processing/image/comparator.hpp"
#include "processing/image/comparison/peak_snr_comparator.hpp"
#include "types/image.hpp"
#include "types/viewpoint.hpp"

template<typename T>
class RadiusRefiner {
public:
    using RenderFunction = std::function<Image<>(const ViewPoint<T> &)>;

    struct RefineResult {
        ViewPoint<T> best_viewpoint;
        int iterations;
    };

    explicit RadiusRefiner(T score_tolerance = 1e-6, T radius_tolerance = 1e-5, const int max_iterations = 50,
                           T initial_step_size = 0.01, T step_reduction_factor = 0.5) :
        score_tolerance_(score_tolerance), radius_tolerance_(radius_tolerance), max_iterations_(max_iterations),
        initial_step_size_(initial_step_size), step_reduction_factor_(step_reduction_factor) {}

    [[nodiscard]] RefineResult refine(const ViewPoint<T> &initial_viewpoint, const Image<> &target,
                                      const RenderFunction &render) const {
        auto [radius, polar, azimuthal] = initial_viewpoint.toSpherical();
        const auto comparator = std::make_shared<processing::image::PeakSNRComparator>();
        RefineResult result{initial_viewpoint, 0};
        T step_size = initial_step_size_ * radius; // Step size relative to current radius
        int stagnant_iterations = 0;

        for (int i = 0; i < max_iterations_; ++i) {
            auto current_viewpoint = ViewPoint<T>::fromSpherical(radius, polar, azimuthal);
            const auto current_image = render(current_viewpoint);
            const T current_score = comparator->compare(target, current_image);
            current_viewpoint.setScore(current_score);

            LOG_INFO("Iteration {}: radius = {:.6f}, score = {:.6f}, step = {:.6f}", i, radius, current_score,
                     step_size);

            if (current_score > result.best_viewpoint.getScore()) {
                result = {current_viewpoint, i + 1};
                stagnant_iterations = 0;
            } else {
                stagnant_iterations++;
                if (stagnant_iterations >= 3) {
                    step_size *= step_reduction_factor_;
                    stagnant_iterations = 0;
                }
            }

            // Try both increasing and decreasing the radius
            T radius_increase = adjustRadius(radius, step_size, true);
            T radius_decrease = adjustRadius(radius, step_size, false);

            auto viewpoint_increase = ViewPoint<T>::fromSpherical(radius_increase, polar, azimuthal);
            auto viewpoint_decrease = ViewPoint<T>::fromSpherical(radius_decrease, polar, azimuthal);

            T score_increase = comparator->compare(target, render(viewpoint_increase));
            T score_decrease = comparator->compare(target, render(viewpoint_decrease));

            if (score_increase > current_score && score_increase > score_decrease) {
                radius = radius_increase;
            } else if (score_decrease > current_score) {
                radius = radius_decrease;
            } else {
                // If no improvement, reduce step size
                step_size *= step_reduction_factor_;
            }

            if (step_size < radius_tolerance_ * radius) {
                LOG_INFO("Step size below tolerance. Stopping.");
                break;
            }
        }

        LOG_INFO("Refinement complete. Best score: {:.6f}, Iterations: {}", result.best_viewpoint.getScore(),
                 result.iterations);
        return result;
    }

private:
    T score_tolerance_;
    T radius_tolerance_;
    int max_iterations_;
    T initial_step_size_;
    T step_reduction_factor_;

    static T adjustRadius(T current_radius, T step_size, bool increase) {
        return increase ? current_radius + step_size : current_radius - step_size;
    }
};

template<typename T>
[[nodiscard]] ViewPoint<T> refineRadius(const ViewPoint<T> &best_viewpoint, const Image<> &target,
                                        const typename RadiusRefiner<T>::RenderFunction &render,
                                        const std::shared_ptr<processing::image::ImageComparator> &comparator) {
    LOG_INFO("Starting radius refinement for viewpoint: {}", best_viewpoint.toString());
    const RadiusRefiner<T> refiner;
    const auto result = refiner.refine(best_viewpoint, target, render, comparator);
    LOG_INFO("Radius refinement complete. New best viewpoint: {}", result.best_viewpoint.toString());
    return result.best_viewpoint;
}

#endif // RADIUS_REFINER_HPP
