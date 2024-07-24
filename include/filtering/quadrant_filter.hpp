// File: core/quadrant_filter.hpp

#ifndef QUADRANT_FILTER_HPP
#define QUADRANT_FILTER_HPP

#include <Eigen/Core>
#include <algorithm>
#include <functional>
#include <vector>
#include "filter.hpp"
#include "types/viewpoint.hpp"

template<typename T = double>
class QuadrantFilter final : public Filter<ViewPoint<T>> {
public:
    std::vector<ViewPoint<T>> filter(const std::vector<ViewPoint<T>> &samples,
                                     std::function<double(const ViewPoint<T> &)> evaluation_function) override;

private:
    std::vector<std::vector<ViewPoint<T>>> partitionIntoQuadrants(const std::vector<ViewPoint<T>> &samples);
    std::vector<ViewPoint<T>> selectBestQuadrants(const std::vector<std::vector<ViewPoint<T>>> &quadrants,
                                                  std::function<double(const ViewPoint<T> &)> evaluation_function);
};

template<typename T>
std::vector<ViewPoint<T>> QuadrantFilter<T>::filter(const std::vector<ViewPoint<T>> &samples,
                                                    std::function<double(const ViewPoint<T> &)> evaluation_function) {
    auto quadrants = partitionIntoQuadrants(samples);
    return selectBestQuadrants(quadrants, evaluation_function);
}

template<typename T>
std::vector<std::vector<ViewPoint<T>>>
QuadrantFilter<T>::partitionIntoQuadrants(const std::vector<ViewPoint<T>> &samples) {
    std::vector<std::vector<ViewPoint<T>>> quadrants(8); // Assuming 8 quadrants for finer granularity

    for (const auto &sample: samples) {
        const auto &position = sample.getPosition();
        int quadrant_index = (position.x() >= 0 ? 1 : 0) + (position.y() >= 0 ? 2 : 0) + (position.z() >= 0 ? 4 : 0);
        quadrants[quadrant_index].push_back(sample);
    }

    return quadrants;
}

template<typename T>
std::vector<ViewPoint<T>>
QuadrantFilter<T>::selectBestQuadrants(const std::vector<std::vector<ViewPoint<T>>> &quadrants,
                                       std::function<double(const ViewPoint<T> &)> evaluation_function) {
    std::vector<std::pair<double, size_t>> quadrant_scores(quadrants.size());

    for (size_t i = 0; i < quadrants.size(); ++i) {
        double total_score = 0;
        for (const auto &sample: quadrants[i]) {
            total_score += evaluation_function(sample);
        }
        quadrant_scores[i] = {total_score / quadrants[i].size(), i};
    }

    std::ranges::sort(quadrant_scores.begin(), quadrant_scores.end(), std::greater<>());

    std::vector<ViewPoint<T>> filtered_samples;
    for (size_t i = 0; i < quadrants.size() / 2; ++i) {
        filtered_samples.insert(filtered_samples.end(), quadrants[quadrant_scores[i].second].begin(),
                                quadrants[quadrant_scores[i].second].end());
    }

    return filtered_samples;
}

#endif // QUADRANT_FILTER_HPP
