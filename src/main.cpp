//
// Created by ayush on 5/21/24.
//

#include <iostream>

#include "../include/tasks.hpp"
#include "../include/config/config.hpp"

using namespace std;

int main() {
    Config::initializeLogging(Config::Paths::logsDirectory + "logfile.log");
    Config::setLoggingLevel(spdlog::level::info);

    srand(43); // Seed for reproducibility

    Tasks tasks;
    std::vector<std::string> objects = {"obj_000020"};
    int test_num = 1;

    try {
        for (const auto &object: objects) {
            tasks.performViewpointEvaluation("task1", object, test_num, "../view_space/5.txt");
            //tasks.performViewpointEvaluation("task2", object, test_num, "../view_space/100.txt");
        }
    } catch (const std::exception &e) {
        std::cerr << "Error encountered: " << e.what() << std::endl;
        return 1; // Return a non-zero value to indicate error
    }

    return 0;
}
