# core/CMakeLists.txt

cmake_minimum_required(VERSION 3.14...3.30)
project(viewpoint_core VERSION 1.0.0 LANGUAGES CXX)

# Include modular CMake configurations
include(cmake/ProjectSettings.cmake)
include(cmake/CompilerSettings.cmake)
include(cmake/Dependencies.cmake)
include(cmake/SourceManagement.cmake)
include(cmake/PrecompiledHeaders.cmake)

# Main library target
add_library(${PROJECT_NAME} STATIC)

# Add sources to the library target
add_project_sources(${PROJECT_NAME})

# Set include directories for the library
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${PROJECT_INCLUDE_DIRS}
)

# Apply precompiled headers to the library
add_precompiled_headers(${PROJECT_NAME})

# Apply compile definitions and link libraries to the library
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PROJECT_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIBRARIES})

# Executable target
add_executable(${PROJECT_NAME}_exe)

# Add the same sources to the executable target
add_project_sources(${PROJECT_NAME}_exe)

# Set include directories for the executable
target_include_directories(${PROJECT_NAME}_exe
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${PROJECT_INCLUDE_DIRS}
)

# Apply precompiled headers to the executable
add_precompiled_headers(${PROJECT_NAME}_exe)

# Apply compile definitions and link libraries to the executable
target_compile_definitions(${PROJECT_NAME}_exe PRIVATE ${PROJECT_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_exe PRIVATE ${PROJECT_LIBRARIES})

# Include after target creation
include(cmake/BuildOptimizations.cmake)
include(cmake/Resources.cmake)

# Set the output name for the executable
set_target_properties(${PROJECT_NAME}_exe PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

message(STATUS "Configuration complete for ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})${CMAKE_CXX_COMPILER_VERSION}")