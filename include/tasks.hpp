//
// Created by ayush on 5/21/24.
//

#ifndef TASKS_HPP
#define TASKS_HPP

#include <string>
#include <opencv2/core.hpp>

class Tasks {
public:
    void execute(const std::string& taskName, const std::string& object_name, int test_num);

private:
    void task1(const std::string& object_name, int test_num);
    // cv::Mat aggregateDescriptors(const std::vector<std::string>& imagePaths);
    // cv::Mat performKMeans(const cv::Mat& data, int numClusters);
};

#endif // TASKS_HPP
