// File: viewpoint/generator.hpp

#ifndef VIEWPOINT_GENERATOR_HPP
#define VIEWPOINT_GENERATOR_HPP

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "viewpoint/provider.hpp"
#include "filtering/heuristic_filter.hpp"
#include "core/view.hpp"

namespace viewpoint {

    class Generator final : public Provider {
    public:
        Generator(int num_samples, int dimensions);

        std::vector<core::View> provision() override;

        void setTargetImage(const cv::Mat &target_image) override;

        void setCameraParameters(const core::Camera::CameraParameters &camera_parameters) override;

        void visualizeSphere(const std::string &window_name) const;

    private:
        int num_samples_;
        int dimensions_;
        cv::Mat target_image_;
        core::Camera::CameraParameters camera_parameters_;
        std::shared_ptr<filtering::HeuristicFilter> filter_chain_;
        double estimated_distance_;

        double estimateDistanceToObject();

        [[nodiscard]] std::vector<std::vector<double> > generateInitialViewpoints(double distance) const;

        [[nodiscard]] std::vector<core::View> convertToViews(const std::vector<std::vector<double> > &samples) const;

        void setupFilters();

        void addHeuristics();
    };

}

#endif // VIEWPOINT_GENERATOR_HPP


/*#ifndef VIEWPOINT_GENERATOR_HPP
#define VIEWPOINT_GENERATOR_HPP

#include <opencv2/opencv.hpp>

#include "viewpoint/provider.hpp"
#include "filtering/heuristic_filter.hpp"

namespace viewpoint {
    class Generator : public Provider {
    public:
        Generator(int num_samples, int dimensions, unsigned int seed);

        std::vector<core::View> provision() override;

        void setTargetImage(const cv::Mat &target_image);

        void setCameraMatrix(const cv::Mat &camera_matrix);

    private:
        int num_samples_;
        int dimensions_;
        unsigned int seed_;
        cv::Mat target_image_;
        cv::Mat camera_matrix_;
        std::shared_ptr<filtering::HeuristicFilter> heuristic_filter_;

        std::pair<float, float> detectAndEstimateScaleDistance();

        std::vector<std::vector<double> > generateInitialViewpoints(float distance);

        std::vector<std::vector<double> > convertToCartesian(const std::vector<std::vector<double> > &spherical_coords);

        std::vector<core::View> convertToViews(const std::vector<std::vector<double> > &samples);

        void addHeuristics();
    };
}

#endif // VIEWPOINT_GENERATOR_HPP*/

