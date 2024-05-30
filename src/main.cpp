//
// Created by ayush on 5/21/24.
//

#include <spdlog/spdlog.h>

#include "../include/task1.hpp"

using namespace std;

enum Task {
    Task_1,
    Task_2,
    Task_3,
    Task_4
};

Task selected_task = Task_1; 

int main() {
    Config::initializeLogging();
    Config::setLoggingLevel(spdlog::level::trace);

    srand(43);


    switch (selected_task) {
        case Task_1:{
            spdlog::info("Task_1: level_1");
            run_level_1();
            break;
        }
        case Task_2:{
            spdlog::info("Task_1: level_2");
            run_level_2();
            break;
        }
        default:
            spdlog::error("Unknown method selected.\n");
            return 1;
    }
    

    

    return 0;
}
