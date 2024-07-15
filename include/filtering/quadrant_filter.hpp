// File: core/quadrant_filter.hpp

#ifndef QUADRANT_FILTER_HPP
#define QUADRANT_FILTER_HPP

#include "filter.hpp"
#include <Eigen/Core>
#include <vector>
#include <functional>

template<typename T>
class QuadrantFilter final : public Filter<T> {
public:
    std::vector<T> filter(const std::vector<T> &samples, std::function<double(const T &)> evaluation_function) override;

private:
    std::vector<std::vector<T> > partitionIntoQuadrants(const std::vector<T> &samples);

    std::vector<T> selectBestQuadrants(const std::vector<std::vector<T> > &quadrants,
                                       std::function<double(const T &)> evaluation_function);
};

template<typename T>
std::vector<T> QuadrantFilter<T>::filter(const std::vector<T> &samples,
                                         std::function<double(const T &)> evaluation_function) {
    auto quadrants = partitionIntoQuadrants(samples);
    return selectBestQuadrants(quadrants, evaluation_function);
}

template<typename T>
std::vector<std::vector<T> > QuadrantFilter<T>::partitionIntoQuadrants(const std::vector<T> &samples) {
    // Placeholder: Implement logic to partition samples into quadrants.
    std::vector<std::vector<T> > quadrants(4); // Assuming 4 quadrants for simplicity.

    for (const auto &sample: samples) {
        const auto &position = sample.getViewPoint().getPosition();
        // Determine the quadrant based on the position coordinates.
        int quadrant_index = (position.x() >= 0 ? 1 : 0) + (position.y() >= 0 ? 2 : 0);
        quadrants[quadrant_index].push_back(sample);
    }

    return quadrants;
}

template<typename T>
std::vector<T> QuadrantFilter<T>::selectBestQuadrants(const std::vector<std::vector<T> > &quadrants,
                                                      std::function<double(const T &)> evaluation_function) {
    std::vector<std::pair<double, size_t> > quadrant_scores(quadrants.size());

    for (size_t i = 0; i < quadrants.size(); ++i) {
        double total_score = 0;
        for (const auto &sample: quadrants[i]) {
            total_score += evaluation_function(sample);
        }
        quadrant_scores[i] = {total_score / quadrants[i].size(), i};
    }

    // Sort quadrants by their average scores.
    std::sort(quadrant_scores.begin(), quadrant_scores.end(), std::greater<>());

    // Select the best quadrants.
    std::vector<T> filtered_samples;
    for (size_t i = 0; i < quadrants.size() / 2; ++i) {
        // Retain top 50% quadrants.
        filtered_samples.insert(filtered_samples.end(),
                                quadrants[quadrant_scores[i].second].begin(),
                                quadrants[quadrant_scores[i].second].end());
    }

    return filtered_samples;
}

#endif // QUADRANT_FILTER_HPP
