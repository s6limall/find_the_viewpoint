cmake_minimum_required(VERSION 3.28)
project(find_the_viewpoint)

set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${JSONCPP_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add executable
add_executable(${PROJECT_NAME} src/task1.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${JSONCPP_LIBRARIES})
