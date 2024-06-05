//
// Created by ayush on 5/21/24.
//

#include "../include/tasks.hpp"
#include "../include/config/config.hpp"

using namespace std;

int main() {
    Config::initializeLogging(Config::Paths::logsDirectory + "logfile.log");
    Config::setLoggingLevel(spdlog::level::trace);

    srand(43);

    Tasks tasks;
    std::vector<std::string> objects = {"obj_000020"};
    int test_num = 5;

    for (const auto& object : objects) {
        tasks.execute("task1", object, test_num);
    }

    return 0;
}
