# Set the minimum CMake version required
cmake_minimum_required(VERSION 3.14...3.30)

# Set CMake policies for modern behavior as early as possible
cmake_policy(SET CMP0074 NEW)  # FindPackage uses <Package>_ROOT variables
cmake_policy(SET CMP0079 NEW)  # target_link_libraries() allow PUBLIC/PRIVATE/INTERFACE
cmake_policy(SET CMP0091 NEW)  # Enable position-independent code for executables

# Integrate ccache for faster builds (if available)
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message(STATUS "ccache found and integrated.")
else ()
    message(STATUS "ccache not found. Building without ccache.")
endif ()

# Define the project name and specify the programming languages
project(find_the_viewpoint LANGUAGES CXX)

# Set the C++ standard and ensure it is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable C language if MPI is found
find_package(MPI QUIET)
if (MPI_FOUND)
    enable_language(C)
endif ()

# Find necessary packages
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(jsoncpp CONFIG REQUIRED)
find_package(Freetype REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

# Set include directories for the project
set(INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${SPDLOG_INCLUDE_DIRS}
)

# Set libraries to link against
set(LIBRARIES
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${FREETYPE_LIBRARIES}
        spdlog::spdlog
        yaml-cpp
        fmt::fmt
)

# Automatically call the script to generate sources.cmake
execute_process(
        COMMAND ${CMAKE_COMMAND} -E env bash ${PROJECT_SOURCE_DIR}/cmake/generate_sources.sh
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE SCRIPT_RESULT
)
if (SCRIPT_RESULT)
    message(FATAL_ERROR "Failed to generate sources.cmake")
else ()
    message(STATUS "sources.cmake generated successfully.")
endif ()

# Include the generated sources.cmake
include(${PROJECT_SOURCE_DIR}/cmake/sources.cmake)

# Add executable target
add_executable(${PROJECT_NAME} ${PROJECT_HEADERS} ${PROJECT_SOURCES})

# Include directories for the target
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

# Link libraries to the executable
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

# Use precompiled headers to reduce compilation time
target_precompile_headers(${PROJECT_NAME} PRIVATE
        <iostream>
        <vector>
        <string>
        <memory>
        <numeric>
        <functional>
        <Eigen/Geometry>
        <Eigen/Dense>
        <pcl/PolygonMesh.h>
        <spdlog/spdlog.h>
        <yaml-cpp/yaml.h>
        <opencv2/core.hpp>
        <opencv2/opencv.hpp>
)

# Custom target to copy configuration.yaml to the output directory
add_custom_target(copy_config ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/configuration.yaml
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        DEPENDS ${PROJECT_SOURCE_DIR}/configuration.yaml
        COMMENT "Copying configuration.yaml to output directory"
)

# Custom target to copy 3d_models directory to the output directory
add_custom_target(copy_3d_models ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${PROJECT_SOURCE_DIR}/3d_models
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/3d_models
        DEPENDS ${PROJECT_SOURCE_DIR}/3d_models
        COMMENT "Copying 3d_models directory to output directory"
)

# Ensure that configuration.yaml and 3d_models are copied after building the main target
add_dependencies(${PROJECT_NAME} copy_config)
add_dependencies(${PROJECT_NAME} copy_3d_models)

# Export compile commands for tools like clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure out-of-source build to keep the source directory clean
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please use a separate build directory.")
endif ()

# Enable compiler warnings for better code quality
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
endif ()

# Set build types to ensure predictable builds
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING "Available build types" FORCE)

# Set output directories for all platforms
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform-specific configurations
if (WIN32)
    # Windows-specific settings
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS) # Suppress Windows-specific warnings
elseif (UNIX)
    # Unix-specific settings
    # Link with pthread for better multi-threading support on Unix
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
endif ()

# Add the tests directory
# add_subdirectory(tests)

# Add the tests directory conditionally
option(BUILD_TESTS "Build the tests" ON)
if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()

# Include testing support
enable_testing()

# Add custom target to clean up binary directories
add_custom_target(clean-all
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}
        COMMENT "Cleaning up all build files and directories."
)

# Enable Unity builds for faster compilation
set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON)
