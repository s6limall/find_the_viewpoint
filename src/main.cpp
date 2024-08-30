// File: main.cpp

#include "common/logging/logger.hpp"
#include "common/timer.hpp"
#include "executor.hpp"


int main() {

    try {
        Timer timer("MAIN");
        Executor::execute();
        timer.stop();
    } catch (const std::exception &e) {
        LOG_ERROR("An error occurred during execution: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}