//
// Created by ayush on 6/5/24.
//

#include "../../include/logging/results_logger.hpp"

#include <fstream>
#include <iostream>

ResultsLogger::TestResult::TestResult(int id, int index, size_t matches, const std::vector<View> &views)
    : test_id(id), selected_view_index(index), good_matches(matches) {
    for (const auto &view: views) {
        poses.push_back(view.pose_6d);
    }
}

void ResultsLogger::addResult(const TestResult &result) {
    results.push_back(result);
}

void ResultsLogger::saveResults(const std::string &filename) const {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error opening the results file: " << filename << std::endl;
        return;
    }
    for (const auto &result: results) {
        fout << "Test ID: " << result.test_id << "\n"
                << "Selected View Index: " << result.selected_view_index << "\n"
                << "Good Matches: " << result.good_matches << "\n"
                << "Selected Views Poses:\n";
        for (const auto &pose: result.poses) {
            fout << pose << "\n";
        }
        fout << "------------------------\n";
    }
    fout.close();
}
