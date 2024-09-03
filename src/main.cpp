// File: main.cpp

#include "common/logging/logger.hpp"
#include "common/state/state.hpp"
#include "common/timer.hpp"
#include "executor.hpp"


int main() {

    try {
        Timer main_timer("MAIN");

        /*const auto objects = common::io::filesByExtension("./3d_models", ".ply");
        LOG_INFO("Found {} .ply files to process", objects.size());*/

        if (config::get("perception.type", "simulator") != "simulator") {
            LOG_WARN("Not using simulator, using: {}; Returning success for deferred processing",
                     config::get("perception.type", "INVALID"));
            return EXIT_SUCCESS;
        }

        const auto objects =
                std::vector<std::filesystem::path>{"./3d_models/" + config::get("object.name", "obj_000020") + ".ply"};

        for (const auto &path: objects) {
            LOG_INFO("Processing mesh: {}", path.string());
        }

        for (const auto &ply_file: objects) {
            Timer iteration_timer("MESH PROCESSING");

            std::string filename = ply_file.stem().string();
            std::string filepath = ply_file.string();

            LOG_INFO("Processing mesh: {} at path: {}", filename, filepath);

            state::set("object.name", filename);
            state::set("paths.mesh", filepath);

            Executor::execute();

            iteration_timer.stop();
            LOG_INFO("Finished processing model: {}", filename);
        }

        main_timer.stop();
        LOG_INFO("All meshes processed successfully");
    } catch (const std::exception &e) {
        LOG_ERROR("An error occurred during execution: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
