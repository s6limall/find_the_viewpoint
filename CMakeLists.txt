cmake_minimum_required(VERSION 3.14...3.30)

# Set CMake policies for modern behavior
cmake_policy(SET CMP0074 NEW)  # FindPackage uses <Package>_ROOT variables
cmake_policy(SET CMP0079 NEW)  # target_link_libraries() allow PUBLIC/PRIVATE/INTERFACE
cmake_policy(SET CMP0091 NEW)  # Enable position-independent code for executables
cmake_policy(SET CMP0102 NEW)  # Evaluate generator expressions during export
cmake_policy(SET CMP0114 NEW)  # Enable imported targets for Apple framework dependencies

# Define the project
project(find_the_viewpoint LANGUAGES CXX)

message(STATUS "Setting up the project ${PROJECT_NAME}")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include modular CMake files
message(STATUS "Including modular CMake files")
include(${CMAKE_SOURCE_DIR}/cmake/CompilerSettings.cmake)    # Compiler and optimization settings
include(${CMAKE_SOURCE_DIR}/cmake/vcpkg.cmake)               # vcpkg setup and integration

# Run the script to generate sources.cmake
execute_process(
        COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_SOURCE_DIR}/cmake/generate_sources.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE SCRIPT_RESULT
)
if (NOT SCRIPT_RESULT EQUAL "0")
    message(FATAL_ERROR "Failed to generate sources.cmake")
endif ()

# Include the generated sources.cmake
include(${CMAKE_SOURCE_DIR}/cmake/sources.cmake)

include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)        # External dependencies
include(${CMAKE_SOURCE_DIR}/cmake/PrecompiledHeaders.cmake)  # Precompiled headers setup

# Set output directories for all platforms
message(STATUS "Setting output directories")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Ensure out-of-source build to keep the source directory clean
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please use a separate build directory.")
endif ()

# Export compile commands for tools like clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add the main executable
message(STATUS "Adding the main executable")
add_executable(${PROJECT_NAME} ${PROJECT_HEADERS} ${PROJECT_SOURCES})

# Include directories and link libraries for the target
message(STATUS "Setting include directories and link libraries")
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

# Set precompiled headers
target_precompile_headers(${PROJECT_NAME} PRIVATE ${PRECOMPILED_HEADERS})

include(${CMAKE_SOURCE_DIR}/cmake/Utilities.cmake)           # Utility functions and platform-specific settings

# Enable Unity builds for faster compilation
message(STATUS "Enabling Unity builds for faster compilation")
set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON)

# Custom target to clean up binary directories
add_custom_target(clean_all
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}
        COMMENT "Cleaning up all build files and directories."
)

# Optionally build tests
option(BUILD_TESTS "Build the tests" ON)
if (BUILD_TESTS)
    message(STATUS "Configuring tests")
    add_subdirectory(tests)
    enable_testing()
endif ()

# Custom target to copy necessary files to the output directory
add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/configuration.yaml
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/3d_models
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/3d_models
        COMMENT "Copying configuration.yaml and 3d_models directory to output directory"
)

message(STATUS "Adding dependency: copy_resources")
add_dependencies(${PROJECT_NAME} copy_resources)

message(STATUS "Configuration complete")
