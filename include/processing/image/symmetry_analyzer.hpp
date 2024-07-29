// File: processing/image/symmetry_analyzer.hpp

#ifndef SYMMETRY_ANALYZER_HPP
#define SYMMETRY_ANALYZER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "common/logging/logger.hpp"
#include "histogram.hpp"
#include "types/image.hpp"

namespace processing::image {

    template<typename T = double>
    class SymmetryAnalyzer {
    public:
        explicit SymmetryAnalyzer(const Image<T> &target_image, int num_bins = 36);

        // Analyze symmetry and return symmetry planes and axes
        void analyze();

        // Get detected symmetry planes
        [[nodiscard]] const std::vector<Eigen::Hyperplane<T, 3>> &getSymmetryPlanes() const;

        // Get detected symmetry axes
        [[nodiscard]] const std::vector<Eigen::Matrix<T, 3, 1>> &getSymmetryAxes() const;

        // String representation of the results
        [[nodiscard]] std::string toString() const;

        // Visualize symmetry planes and axes
        void visualize() const;

    private:
        const Image<T> &target_image_;
        int num_orientation_bins_;
        std::vector<int> orientation_histogram_;
        std::vector<Eigen::Hyperplane<T, 3>> symmetry_planes_;
        std::vector<Eigen::Matrix<T, 3, 1>> symmetry_axes_;

        // Compute histogram of keypoint orientations
        void computeOrientationHistogram();

        // Detect symmetry planes and axes based on the orientation histogram
        void detectSymmetryPlanesAndAxes();
    };

    // Function Definitions

    template<typename T>
    SymmetryAnalyzer<T>::SymmetryAnalyzer(const Image<T> &target_image, int num_bins) :
        target_image_(target_image), num_orientation_bins_(num_bins) {
        computeOrientationHistogram();
    }

    template<typename T>
    void SymmetryAnalyzer<T>::analyze() {
        detectSymmetryPlanesAndAxes();
    }

    template<typename T>
    const std::vector<Eigen::Hyperplane<T, 3>> &SymmetryAnalyzer<T>::getSymmetryPlanes() const {
        return symmetry_planes_;
    }

    template<typename T>
    const std::vector<Eigen::Matrix<T, 3, 1>> &SymmetryAnalyzer<T>::getSymmetryAxes() const {
        return symmetry_axes_;
    }

    template<typename T>
    std::string SymmetryAnalyzer<T>::toString() const {
        std::string result = "Symmetry Planes:\n";
        for (const auto &plane: symmetry_planes_) {
            result += "Plane: Normal = (" + std::to_string(plane.normal().x()) + ", " +
                      std::to_string(plane.normal().y()) + ", " + std::to_string(plane.normal().z()) + ")\n";
        }

        result += "Symmetry Axes:\n";
        for (const auto &axis: symmetry_axes_) {
            result += "Axis: Direction = (" + std::to_string(axis.x()) + ", " + std::to_string(axis.y()) + ", " +
                      std::to_string(axis.z()) + ")\n";
        }
        return result;
    }

    template<typename T>
    void SymmetryAnalyzer<T>::visualize() const {
        cv::Mat visualization_image = target_image_.getImage().clone();
        cv::Point center(visualization_image.cols / 2, visualization_image.rows / 2);
        const int line_length = 100;

        // Draw symmetry planes
        for (const auto &plane: symmetry_planes_) {
            const auto normal = plane.normal();
            cv::Point end(center.x + static_cast<int>(normal.x() * line_length),
                          center.y + static_cast<int>(normal.y() * line_length));
            cv::line(visualization_image, center, end, cv::Scalar(0, 255, 0), 2);
        }

        // Draw symmetry axes
        for (const auto &axis: symmetry_axes_) {
            const auto direction = axis;
            cv::Point end(center.x + static_cast<int>(direction.x() * line_length),
                          center.y + static_cast<int>(direction.y() * line_length));
            cv::line(visualization_image, center, end, cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Symmetry Visualization", visualization_image);
        cv::waitKey(0); // Wait for a key press to close the window
    }

    template<typename T>
    void SymmetryAnalyzer<T>::computeOrientationHistogram() {
        orientation_histogram_.resize(num_orientation_bins_, 0);
        const auto &keypoints = target_image_.getKeypoints();
        for (const auto &kp: keypoints) {
            float angle = kp.angle;
            int bin = static_cast<int>(angle / 360.0 * num_orientation_bins_);
            ++orientation_histogram_[std::clamp(bin, 0, num_orientation_bins_ - 1)];
        }
    }

    template<typename T>
    void SymmetryAnalyzer<T>::detectSymmetryPlanesAndAxes() {
        // Determine the threshold for significant symmetry based on histogram peaks
        const int max_count = *std::max_element(orientation_histogram_.begin(), orientation_histogram_.end());
        const double threshold_factor = 0.5; // Keeping the most significant 50% of directions
        const int significant_threshold = static_cast<int>(max_count * threshold_factor);

        for (int i = 0; i < num_orientation_bins_; ++i) {
            if (orientation_histogram_[i] >= significant_threshold) {
                float angle = (i + 0.5f) * 360.0f / num_orientation_bins_;
                float radians = angle * static_cast<float>(CV_PI) / 180.0f;
                Eigen::Matrix<T, 3, 1> direction(std::cos(radians), std::sin(radians), 0);
                symmetry_axes_.push_back(direction);
                symmetry_planes_.emplace_back(direction, 0);
            }
        }
    }

} // namespace processing::image

#endif // SYMMETRY_ANALYZER_HPP
