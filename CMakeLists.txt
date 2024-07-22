cmake_minimum_required(VERSION 3.28)
project(run_find_the_view LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable C language if required by MPI
find_package(MPI QUIET)
if (MPI_FOUND)
    enable_language(C)
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find packages
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(Freetype REQUIRED)
find_package(spdlog REQUIRED)
find_package(Torch REQUIRED PATHS ${TORCH_ROOT})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "PCL version: ${PCL_VERSION}")
message(STATUS "PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")
message(STATUS "Torch_INCLUDE_DIR: ${TORCH_INCLUDE_DIR}")
message(STATUS "Torch_LIBRARIES: ${TORCH_LIBRARIES}")

# Collect include directories and libraries
set(INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${JSONCPP_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${SPDLOG_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIR}
)
set(LIBRARIES
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${FREETYPE_LIBRARIES}
        ${TORCH_LIBRARIES}
        spdlog::spdlog
)

# Include directories
include_directories(${INCLUDE_DIRS})

# Collect source files and headers
file(GLOB_RECURSE PROJECT_HEADERS CONFIGURE_DEPENDS include/*.hpp)
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS src/*.cpp)

# Add executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

cmake_policy(SET CMP0144 NEW)