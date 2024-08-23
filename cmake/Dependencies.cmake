# File: cmake/Dependencies.cmake

# Find and configure required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui xfeatures2d quality viz)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(fmt REQUIRED)


# Aggregate include directories
set(PROJECT_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${spdlog_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${JSONCPP_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
)


# Aggregate libraries
set(PROJECT_LIBRARIES
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp
        jsoncpp
        fmt::fmt
)