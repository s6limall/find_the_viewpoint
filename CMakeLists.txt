cmake_minimum_required(VERSION 3.28)
project(find_the_viewpoint LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enable C language if required by MPI
find_package(MPI QUIET)
if (MPI_FOUND)
    enable_language(C)
endif()

# Find packages
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(jsoncpp CONFIG REQUIRED)
find_package(Freetype REQUIRED)
find_package(spdlog REQUIRED)

# Collect include directories and libraries
set(INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${JSONCPP_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${SPDLOG_INCLUDE_DIRS}
)
set(LIBRARIES
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${FREETYPE_LIBRARIES}
        spdlog::spdlog
)
# Include directories
include_directories(${INCLUDE_DIRS})
# Collect source files and headers
file(GLOB_RECURSE PROJECT_HEADERS CONFIGURE_DEPENDS include/*.hpp)
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS src/*.cpp)
# Add executable
add_executable(${PROJECT_NAME} ${PROJECT_HEADERS} ${PROJECT_SOURCES})
# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
# Enable parallel builds
cmake_policy(SET CMP0144 NEW)
