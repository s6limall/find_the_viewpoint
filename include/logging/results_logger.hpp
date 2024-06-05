//
// Created by ayush on 6/5/24.
//

#ifndef RESULTS_LOGGER_HPP
#define RESULTS_LOGGER_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>

#include "../core/view.hpp"

class ResultsLogger {
public:
    struct TestResult {
        int test_id;
        int selected_view_index;
        size_t good_matches;
        std::vector<Eigen::Matrix4d> poses;

        TestResult(int id, int index, size_t matches, const std::vector<View>& views);
    };

    std::vector<TestResult> results;

    void addResult(const TestResult& result);
    void saveResults(const std::string& filename) const;
};

#endif // RESULTS_LOGGER_HPP
