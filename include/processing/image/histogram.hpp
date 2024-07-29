// File: processing/image/histogram.hpp

#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <Eigen/Dense>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>
#include "common/logging/logger.hpp"
#include "types/image.hpp"

namespace processing::image {

    template<typename T = double>
    class Histogram {
    public:
        // Optional parameters for number of bins and grid size
        explicit Histogram(const Image<T> &target_image, const int num_bins = 36, const int grid_size = 10) :
            target_image_(target_image), num_bins_(num_bins > 0 ? num_bins : 36),
            grid_size_(grid_size > 0 ? grid_size : 10) {}

        // Getters
        [[nodiscard]] const std::vector<int> &getOrientationHistogram() const {
            if (!orientation_histogram_) {
                computeOrientationHistogram();
            }
            return *orientation_histogram_;
        }

        [[nodiscard]] const std::vector<std::vector<int>> &getKeypointDensity() const {
            if (!keypoint_density_) {
                analyzeKeypointDensity();
            }
            return *keypoint_density_;
        }

        // Fluent API for setting parameters
        Histogram &setNumBins(const int num_bins) {
            num_bins_ = (num_bins > 0) ? num_bins : 36;
            orientation_histogram_.reset();
            return *this;
        }

        Histogram &setGridSize(const int grid_size) {
            grid_size_ = (grid_size > 0) ? grid_size : 10;
            keypoint_density_.reset();
            return *this;
        }

        // String representation
        [[nodiscard]] std::string toString() const {
            std::string result = "Orientation Histogram:\n";
            for (int i = 0; i < num_bins_; ++i) {
                result += "Bin " + std::to_string(i) + ": " + std::to_string(getOrientationHistogram()[i]) +
                          " keypoints\n";
            }

            result += "Keypoint Density:\n";
            for (int y = 0; y < static_cast<int>(getKeypointDensity().size()); ++y) {
                for (int x = 0; x < static_cast<int>(getKeypointDensity()[y].size()); ++x) {
                    result += "Cell (" + std::to_string(x) + ", " + std::to_string(y) +
                              "): " + std::to_string(getKeypointDensity()[y][x]) + " keypoints\n";
                }
            }
            return result;
        }

        // Visualization function
        void visualize() const {
            visualizeOrientationHistogram();
            visualizeKeypointDensity();
        }

    private:
        const Image<T> &target_image_;
        int num_bins_;
        int grid_size_;
        mutable std::optional<std::vector<int>> orientation_histogram_;
        mutable std::optional<std::vector<std::vector<int>>> keypoint_density_;

        // Compute histogram of keypoint orientations
        void computeOrientationHistogram() const {
            orientation_histogram_ = std::vector<int>(num_bins_, 0);
            for (const auto &kp: target_image_.getKeypoints()) {
                const float angle = kp.angle;
                const int bin = static_cast<int>(angle / 360.0 * num_bins_);
                if (bin >= 0 && bin < num_bins_) {
                    ++(*orientation_histogram_)[bin];
                }
            }
        }

        // Analyze keypoint density distribution
        void analyzeKeypointDensity() const {
            const int cell_width = target_image_.getImage().cols / grid_size_;
            const int cell_height = target_image_.getImage().rows / grid_size_;
            keypoint_density_ = std::vector<std::vector<int>>(grid_size_, std::vector<int>(grid_size_, 0));

            for (const auto &kp: target_image_.getKeypoints()) {
                const int cell_x = static_cast<int>(kp.pt.x / static_cast<float>(cell_width));
                const int cell_y = static_cast<int>(kp.pt.y / static_cast<float>(cell_height));
                if (cell_x >= 0 && cell_x < grid_size_ && cell_y >= 0 && cell_y < grid_size_) {
                    ++(*keypoint_density_)[cell_y][cell_x];
                }
            }
        }

        // Visualize orientation histogram
        void visualizeOrientationHistogram() const {
            constexpr int hist_width = 512;
            constexpr int hist_height = 400;
            const int bin_width = cvRound(static_cast<double>(hist_width) / num_bins_);

            cv::Mat hist_image(hist_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));
            std::vector<int> histogram = getOrientationHistogram();

            const int max_count = *std::ranges::max_element(histogram.begin(), histogram.end());
            for (int &value: histogram) {
                value = static_cast<int>((static_cast<double>(value) / max_count) * hist_height);
            }

            for (int i = 0; i < num_bins_; ++i) {
                cv::rectangle(hist_image, cv::Point(bin_width * i, hist_height),
                              cv::Point(bin_width * (i + 1), hist_height - histogram[i]), cv::Scalar(255, 0, 0),
                              cv::FILLED);
            }

            cv::imshow("Orientation Histogram", hist_image);
            cv::waitKey(0);
        }

        // Visualize keypoint density
        void visualizeKeypointDensity() const {
            const int cell_width = target_image_.getImage().cols / grid_size_;
            const int cell_height = target_image_.getImage().rows / grid_size_;
            cv::Mat density_image = target_image_.getImage().clone();

            for (int y = 0; y < static_cast<int>(getKeypointDensity().size()); ++y) {
                for (int x = 0; x < static_cast<int>(getKeypointDensity()[y].size()); ++x) {
                    const int keypoints_in_cell = getKeypointDensity()[y][x];
                    if (keypoints_in_cell > 0) {
                        cv::rectangle(density_image, cv::Point(x * cell_width, y * cell_height),
                                      cv::Point((x + 1) * cell_width, (y + 1) * cell_height), cv::Scalar(0, 0, 255),
                                      cv::FILLED);
                    }
                }
            }

            cv::imshow("Keypoint Density", density_image);
            cv::waitKey(0);
        }
    };

} // namespace processing::image

#endif // HISTOGRAM_HPP
