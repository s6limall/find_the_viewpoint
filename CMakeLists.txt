cmake_minimum_required(VERSION 3.14...3.30)

project(find_the_viewpoint VERSION 1.0.0 LANGUAGES CXX)

# Include modular CMake configurations
include(cmake/ProjectSettings.cmake)
include(cmake/CompilerSettings.cmake)
include(cmake/Dependencies.cmake)
include(cmake/SourceManagement.cmake)
include(cmake/PrecompiledHeaders.cmake)

# Main library target
add_library(${PROJECT_NAME}_lib STATIC)  # Library target

# Add sources to the library target
add_project_sources(${PROJECT_NAME}_lib)

# Set include directories for the library
target_include_directories(${PROJECT_NAME}_lib
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${PROJECT_INCLUDE_DIRS}
)

# Apply precompiled headers to the library
add_precompiled_headers(${PROJECT_NAME}_lib)

# Apply compile definitions and link libraries to the library
target_compile_definitions(${PROJECT_NAME}_lib PRIVATE ${PROJECT_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_lib PRIVATE ${PROJECT_LIBRARIES})

# Create the executable that depends on the core library
add_executable(${PROJECT_NAME} src/main.cpp)  # Assuming main.cpp is your entry point

# Link the core library to the executable
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib ${PROJECT_LIBRARIES})

# Set include directories for the executable
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include  # Include generated headers if any
)

# Export the core library target for usage in other modules (optional)
export(TARGETS ${PROJECT_NAME}_lib FILE coreTargets.cmake)

# Install headers, library, and executable for external usage
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_lib EXPORT coreTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)
install(EXPORT coreTargets FILE coreTargets.cmake DESTINATION lib/cmake/core)

# Include after target creation
include(cmake/BuildOptimizations.cmake)
include(cmake/Resources.cmake)

message(STATUS "Configuration complete for ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
