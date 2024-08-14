// File: optimization/radius_refiner.hpp

#ifndef RADIUS_REFINER_HPP
#define RADIUS_REFINER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <opencv2/opencv.hpp>
#include "common/logging/logger.hpp"
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

    explicit RadiusRefiner(T area_tolerance = 1e-6, T radius_tolerance = 1e-5, const int max_iterations = 50,
                           const int max_pyramid_level = 3) :
        area_tolerance_(area_tolerance), radius_tolerance_(radius_tolerance), max_iterations_(max_iterations),
        max_pyramid_level_(max_pyramid_level) {}

    [[nodiscard]] RefineResult refine(const ViewPoint<T> &initial_viewpoint, const Image<> &target,
                                      const RenderFunction &render) const {
        auto [radius, polar, azimuthal] = initial_viewpoint.toSpherical();
        RefineResult result{initial_viewpoint, 0};

        for (int level = max_pyramid_level_; level >= 0; --level) {
            T step_size = initial_step_size_ / (1 << level); // Decrease step size at finer levels
            auto downsampled_target = downsampleImage(target, level);
            auto downsampled_current = renderImageAtRadius(radius, level, polar, azimuthal, render);

            radius = adjustRadius(downsampled_target, downsampled_current, radius, step_size);
            result.best_viewpoint = ViewPoint<T>::fromSpherical(radius, polar, azimuthal);
            ++result.iterations;
        }

        LOG_INFO("Refinement complete. Best radius: {:.6f}, Iterations: {}",
                 std::get<0>(result.best_viewpoint.toSpherical()), result.iterations);
        return result;
    }

private:
    T area_tolerance_;
    T radius_tolerance_;
    int max_iterations_;
    int max_pyramid_level_;
    T initial_step_size_ = 0.01;

    static T calculateObjectArea(const Image<> &image) {
        // Use contour detection to calculate the area of the object in the image
        cv::Mat gray, edges;
        cv::cvtColor(image.getImage(), gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        T area = 0.0;
        for (const auto &contour: contours) {
            area += cv::contourArea(contour);
        }

        return area;
    }

    [[nodiscard]] static Image<> downsampleImage(const Image<> &image, const int level) {
        // Downsample the image based on the pyramid level
        cv::Mat downsampled = image.getImage();
        for (int i = 0; i < level; ++i) {
            cv::pyrDown(downsampled, downsampled);
        }
        return Image<>(downsampled);
    }

    Image<> renderImageAtRadius(T radius, const int level, T polar, T azimuthal, const RenderFunction &render) const {
        auto viewpoint = ViewPoint<T>::fromSpherical(radius, polar, azimuthal);
        return downsampleImage(render(viewpoint), level);
    }

    T adjustRadius(const Image<> &target, const Image<> &current, T current_radius, T step_size) const {
        T current_area = calculateObjectArea(current);
        T target_area = calculateObjectArea(target);

        if (std::abs(current_area - target_area) < area_tolerance_) {
            return current_radius; // Areas are close enough
        }

        if (current_area > target_area) {
            return current_radius + step_size; // Move outward
        } else {
            return current_radius - step_size; // Move inward
        }
    }
};

template<typename T>
[[nodiscard]] ViewPoint<T> refineRadius(const ViewPoint<T> &best_viewpoint, const Image<> &target,
                                        const typename RadiusRefiner<T>::RenderFunction &render) {
    LOG_INFO("Starting radius refinement for viewpoint: {}", best_viewpoint.toString());
    const RadiusRefiner<T> refiner;
    const auto result = refiner.refine(best_viewpoint, target, render);
    LOG_INFO("Radius refinement complete. New best viewpoint: {}", result.best_viewpoint.toString());
    return result.best_viewpoint;
}

#endif // RADIUS_REFINER_HPP
