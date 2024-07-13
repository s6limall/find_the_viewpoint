/*
// File: viewpoint/generator.hpp

#ifndef VIEWPOINT_GENERATOR_HPP
#define VIEWPOINT_GENERATOR_HPP

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "core/view.hpp"
#include "viewpoint/provider.hpp"
#include "processing/vision/distance_estimator.hpp"
// #include "filtering/heuristic_filter.hpp"
#include "filtering/heuristics/distance_heuristic.hpp"
#include "filtering/heuristics/similarity_heuristic.hpp"

namespace viewpoint {

    class Generator final : public Provider {
    public:
        Generator(int num_samples, int dimensions);

        std::vector<core::View> provision() override;

        void setTargetImage(const cv::Mat &target_image) override;

        void setCameraIntrinsics(const core::Camera::Intrinsics &camera_intrinsics) override;

        void visualizeSphere(const std::string &window_name) const;

    private:
        int num_samples_;
        int dimensions_;
        cv::Mat target_image_;
        core::Camera::Intrinsics camera_intrinsics_;
        // std::shared_ptr<filtering::HeuristicFilter> filter_chain_;
        double estimated_distance_;

        double estimateDistanceToObject();

        [[nodiscard]] std::vector<std::vector<double> > generateInitialViewpoints(double distance) const;

        [[nodiscard]] std::vector<core::View> convertToViews(const std::vector<std::vector<double> > &samples) const;

        void setupFilters();

        void addHeuristics() const;
    };

}

#endif // VIEWPOINT_GENERATOR_HPP
*/
