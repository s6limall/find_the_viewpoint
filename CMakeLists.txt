cmake_minimum_required(VERSION 3.14...3.30)

project(find_the_viewpoint VERSION 1.0.0 LANGUAGES CXX)

# Setup vcpkg
include(cmake/vcpkg.cmake)

# Include modular CMake configurations
include(cmake/ProjectSettings.cmake)
include(cmake/CompilerSettings.cmake)
include(cmake/Dependencies.cmake)
include(cmake/SourceManagement.cmake)
include(cmake/PrecompiledHeaders.cmake)

# Main target
add_executable(${PROJECT_NAME})

# Add sources to the target
add_project_sources(${PROJECT_NAME})

# Set include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_INCLUDE_DIRS})

# Apply precompiled headers
add_precompiled_headers(${PROJECT_NAME})

# Apply compile definitions and link libraries
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PROJECT_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIBRARIES})

# Include after target creation
include(cmake/BuildOptimizations.cmake)
include(cmake/Resources.cmake)

message(STATUS "Configuration complete for ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
