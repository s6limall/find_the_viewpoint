//
// Created by ayush on 5/21/24.
//

#include <spdlog/spdlog.h>

#include "../include/task1.hpp"
#include "../include/task2.hpp"

using namespace std;

enum Level {
    Level_1,
    Level_2,
    Level_3,
};

Level selected_task = Level_3;

int main() {
    Config::initializeLogging();
    Config::setLoggingLevel(spdlog::level::trace);

    srand(43);



    task2::run_level_3();
    /*
    switch (selected_task) {
        case Level_1:{
            spdlog::info("Task_1: level_1");
            task1::run_level_1();
            break;
        }
        case Level_2:{
            spdlog::info("Task_1: level_2");
            task1::run_level_2();
            break;
        }
        case Level_3:{
            spdlog::info("Task_2: level_3");
            task2::run_level_3();
            break;
        }
        default:
            spdlog::error("Unknown method selected.\n");
            return 1;
    }
    */

    return 0;
}
