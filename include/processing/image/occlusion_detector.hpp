// File: processing/image/occlusion_detector.hpp

#ifndef OCCLUSION_DETECTOR_HPP
#define OCCLUSION_DETECTOR_HPP

#include <Eigen/Dense>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include "common/logging/logger.hpp"
#include "types/image.hpp"

namespace processing::image {

    template<typename T = double>
    class OcclusionDetector {
    public:
        // Constructor
        explicit OcclusionDetector(const Image<T> &target_image, int grid_rows = 10, int grid_cols = 10);

        // Analyze occlusions and return occluded regions
        void detectOcclusions();

        // Get occluded regions
        [[nodiscard]] const std::vector<std::vector<bool>> &getOccludedRegions() const;

        // String representation of the results
        [[nodiscard]] std::string toString() const;

        // Visualize occluded regions and keypoint density
        void visualize() const;

    private:
        const Image<T> &target_image_;
        int grid_rows_;
        int grid_cols_;
        std::vector<std::vector<int>> keypoint_density_;
        std::vector<std::vector<bool>> occluded_regions_;

        // Compute keypoint density in the image grid
        void computeKeypointDensity();

        // Identify regions with unusually low feature density
        void identifyOccludedRegions();

        // Generate a mask for the object region
        cv::Mat generateObjectMask() const;
    };

    // Function Definitions

    template<typename T>
    OcclusionDetector<T>::OcclusionDetector(const Image<T> &target_image, int grid_rows, int grid_cols) :
        target_image_(target_image), grid_rows_(grid_rows), grid_cols_(grid_cols) {
        detectOcclusions();
    }

    template<typename T>
    void OcclusionDetector<T>::detectOcclusions() {
        try {
            computeKeypointDensity();
            identifyOccludedRegions();
        } catch (const std::exception &e) {
            LOG_ERROR("Error during occlusion detection: {}", e.what());
            throw;
        }
    }

    template<typename T>
    const std::vector<std::vector<bool>> &OcclusionDetector<T>::getOccludedRegions() const {
        return occluded_regions_;
    }

    template<typename T>
    std::string OcclusionDetector<T>::toString() const {
        std::ostringstream result;
        result << "Occluded Regions:\n";
        for (int row = 0; row < grid_rows_; ++row) {
            for (int col = 0; col < grid_cols_; ++col) {
                if (occluded_regions_[row][col]) {
                    LOG_TRACE("Cell ({}, {}) is occluded!", row, col);
                    result << "Cell (" << row << ", " << col << ") is occluded!\n";
                }
            }
        }

        LOG_TRACE("Occlusion Detection Results: \n{}", result.str());

        return result.str();
    }

    template<typename T>
    void OcclusionDetector<T>::visualize() const {
        cv::Mat visualization_image = target_image_.getImage().clone();
        const int cell_width = target_image_.getImage().cols / grid_cols_;
        const int cell_height = target_image_.getImage().rows / grid_rows_;

        // Draw grid and occluded regions
        for (int row = 0; row < grid_rows_; ++row) {
            for (int col = 0; col < grid_cols_; ++col) {
                cv::Rect cell_rect(col * cell_width, row * cell_height, cell_width, cell_height);
                cv::rectangle(visualization_image, cell_rect, cv::Scalar(255, 0, 0), 1); // Draw grid cells in blue
                if (occluded_regions_[row][col]) {
                    cv::rectangle(visualization_image, cell_rect, cv::Scalar(0, 0, 255), 2); // Mark occlusions in red
                }
                cv::putText(visualization_image, std::to_string(keypoint_density_[row][col]),
                            cv::Point(col * cell_width + cell_width / 2, row * cell_height + cell_height / 2),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
            }
        }

        // Draw keypoints
        cv::Mat keypoint_image;
        cv::drawKeypoints(target_image_.getImage(), target_image_.getKeypoints(), keypoint_image,
                          cv::Scalar(0, 255, 0));

        // Combine both images
        cv::addWeighted(visualization_image, 0.5, keypoint_image, 0.5, 0, visualization_image);

        cv::imshow("Occluded Regions", visualization_image);
        cv::waitKey(0); // Wait for a key press to close the window
    }

    template<typename T>
    void OcclusionDetector<T>::computeKeypointDensity() {
        keypoint_density_.resize(grid_rows_, std::vector<int>(grid_cols_, 0));
        const auto &keypoints = target_image_.getKeypoints();
        const int cell_width = target_image_.getImage().cols / grid_cols_;
        const int cell_height = target_image_.getImage().rows / grid_rows_;

        for (const auto &keypoint: keypoints) {
            const int col = static_cast<int>(keypoint.pt.x / cell_width);
            const int row = static_cast<int>(keypoint.pt.y / cell_height);
            if (col >= 0 && col < grid_cols_ && row >= 0 && row < grid_rows_) {
                ++keypoint_density_[row][col];
            }
        }
    }

    template<typename T>
    void OcclusionDetector<T>::identifyOccludedRegions() {
        occluded_regions_.resize(grid_rows_, std::vector<bool>(grid_cols_, false));
        std::vector<int> densities;

        for (const auto &row: keypoint_density_) {
            densities.insert(densities.end(), row.begin(), row.end());
        }

        const double mean_density = std::accumulate(densities.begin(), densities.end(), 0.0) / densities.size();
        const double stddev_density =
                std::sqrt(std::accumulate(densities.begin(), densities.end(), 0.0,
                                          [mean_density](double sum, int density) {
                                              return sum + (density - mean_density) * (density - mean_density);
                                          }) /
                          densities.size());

        // Threshold based on mean density minus a factor of standard deviation
        constexpr double threshold_factor = 0.5; // TODO: Adjust later
        const double occlusion_threshold = mean_density - threshold_factor * stddev_density;

        LOG_DEBUG("Mean Density: {}", mean_density);
        LOG_DEBUG("Standard Deviation of Density: {}", stddev_density);
        LOG_DEBUG("Occlusion Threshold: {}", occlusion_threshold);

        for (int row = 0; row < grid_rows_; ++row) {
            for (int col = 0; col < grid_cols_; ++col) {
                if (keypoint_density_[row][col] == 0) {
                    occluded_regions_[row][col] = true; // TODO: Remove == 0 if necessary
                } else if (keypoint_density_[row][col] < occlusion_threshold) {
                    occluded_regions_[row][col] = true;
                }
            }
        }
    }

    template<typename T>
    cv::Mat OcclusionDetector<T>::generateObjectMask() const {
        // Create a mask for the object region
        cv::Mat mask = cv::Mat::zeros(target_image_.getImage().size(), CV_8UC1);
        const auto &keypoints = target_image_.getKeypoints();
        for (const auto &kp: keypoints) {
            mask.at<uchar>(cv::Point(kp.pt.x, kp.pt.y)) = 255;
        }
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 5);
        return mask;
    }

} // namespace processing::image

#endif // OCCLUSION_DETECTOR_HPP
