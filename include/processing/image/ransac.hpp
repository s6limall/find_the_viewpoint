// File: processing/image/ransac.hpp

#ifndef PROCESSING_IMAGE_RANSAC_HPP
#define PROCESSING_IMAGE_RANSAC_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace processing::image {

    /**
     * @brief RANSAC (Random Sample Consensus) class template for robust model fitting.
     *
     * This class provides a generic implementation of the RANSAC algorithm, allowing for robust
     * fitting of models to data points in the presence of outliers. It is designed to be flexible
     * and reusable with different types of data points and models.
     *
     * @tparam PointType Type of data points (e.g., Eigen::Vector3d for 3D points).
     * @tparam ModelType Type of the model to be fitted (e.g., Eigen::Hyperplane<double, 3> for a plane).
     */
    template<typename PointType, typename ModelType>
    class RANSAC {
    public:
        struct Options {
            constexpr static int max_iterations = 1000;
            constexpr static double distance_threshold = 0.01;
            constexpr static int min_inliers = 50;
            constexpr static double confidence = 0.99; // Confidence level for early termination
        };

        struct Result {
            ModelType model{};
            std::vector<int> inliers{};
            bool success{false};
            double inlier_ratio{0.0};
        };

        /**
         * @brief Constructs a RANSAC object with specified model fitter and distance calculator.
         *
         * @param model_fitter Function to fit a model to a set of points.
         * @param distance_calculator Function to calculate the distance from a point to a model.
         * @param options Configuration options for the RANSAC algorithm.
         */
        RANSAC(const std::function<ModelType(const std::vector<PointType> &)> &model_fitter,
               const std::function<double(const PointType &, const ModelType &)> &distance_calculator,
               const Options &options = Options{}) noexcept;

        [[nodiscard]] std::optional<Result> findDominantPlane(const std::vector<PointType> &points) const noexcept;

    private:
        const std::function<ModelType(const std::vector<PointType> &)> model_fitter_;
        const std::function<double(const PointType &, const ModelType &)> distance_calculator_;
        const Options options_;

        /**
         * @brief Calculates the inliers for a given model.
         *
         * @param points Vector of data points.
         * @param model Fitted model.
         * @return Vector of indices of inlier points.
         */
        [[nodiscard]] std::vector<int> getInliers(const std::vector<PointType> &points,
                                                  const ModelType &model) const noexcept;

        /**
         * @brief Generates unique random indices for selecting sample points.
         *
         * @tparam RNG Random number generator type.
         * @param count Number of unique indices to generate.
         * @param max_value Maximum value for the indices.
         * @param gen Random number generator.
         * @param dis Uniform integer distribution.
         * @return Vector of unique random indices.
         */
        template<typename RNG>
        [[nodiscard]] static std::vector<int> generateUniqueIndices(int count, int max_value, RNG &gen,
                                                                    std::uniform_int_distribution<> &dis) noexcept;
    };

    // Constructor definition
    template<typename PointType, typename ModelType>
    RANSAC<PointType, ModelType>::RANSAC(
            const std::function<ModelType(const std::vector<PointType> &)> &model_fitter,
            const std::function<double(const PointType &, const ModelType &)> &distance_calculator,
            const Options &options) noexcept :
        model_fitter_(model_fitter), distance_calculator_(distance_calculator), options_(options) {}

    template<typename PointType, typename ModelType>
    std::optional<typename RANSAC<PointType, ModelType>::Result>
    RANSAC<PointType, ModelType>::findDominantPlane(const std::vector<PointType> &points) const noexcept {
        if (points.size() < 3) {
            return std::nullopt;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, points.size() - 1);

        Result best_result;
        int best_inlier_count = 0;
        double best_inlier_ratio = 0.0;

        for (int iter = 0; iter < options_.max_iterations; ++iter) {
            const auto indices = generateUniqueIndices(3, points.size(), gen, dis);

            std::vector<PointType> sample_points(3);
            std::transform(indices.begin(), indices.end(), sample_points.begin(),
                           [&points](int index) { return points[index]; });

            const auto model = model_fitter_(sample_points);
            const auto inliers = getInliers(points, model);
            const double inlier_ratio = static_cast<double>(inliers.size()) / points.size();

            if (inliers.size() > best_inlier_count) {
                best_inlier_count = inliers.size();
                best_result.model = model;
                best_result.inliers = inliers;
                best_result.success = true;
                best_inlier_ratio = inlier_ratio;

                if (1.0 - std::pow(inlier_ratio, 3) >= options_.confidence) {
                    break;
                }
            }
        }

        if (best_result.success) {
            best_result.inlier_ratio = best_inlier_ratio;
            return best_result;
        }

        return std::nullopt;
    }

    template<typename PointType, typename ModelType>
    std::vector<int> RANSAC<PointType, ModelType>::getInliers(const std::vector<PointType> &points,
                                                              const ModelType &model) const noexcept {
        std::vector<int> inliers;
        inliers.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            if (distance_calculator_(points[i], model) < options_.distance_threshold) {
                inliers.push_back(static_cast<int>(i));
            }
        }
        return inliers;
    }

    // generateUniqueIndices definition
    template<typename PointType, typename ModelType>
    template<typename RNG>
    std::vector<int>
    RANSAC<PointType, ModelType>::generateUniqueIndices(const int count, int max_value, RNG &gen,
                                                        std::uniform_int_distribution<> &dis) noexcept {
        std::vector<int> indices;
        indices.reserve(count);
        while (indices.size() < count) {
            if (int idx = dis(gen); std::find(indices.begin(), indices.end(), idx) == indices.end()) {
                indices.push_back(idx);
            }
        }
        return indices;
    }

} // namespace processing::image

#endif // PROCESSING_IMAGE_RANSAC_HPP
