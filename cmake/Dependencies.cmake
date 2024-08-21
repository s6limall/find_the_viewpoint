# File: cmake/Dependencies.cmake

# Dependencies.cmake

# Optionally prefer system packages
option(PREFER_SYSTEM_PACKAGES "Prefer system packages over other methods if available" ON)

# Find and configure required packages

# OpenCV with Contrib modules
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui xfeatures2d quality)

# Eigen3 library
find_package(Eigen3 REQUIRED NO_MODULE)

# Logging library
find_package(spdlog REQUIRED)

# YAML and JSON libraries
find_package(yaml-cpp REQUIRED)
find_package(jsoncpp REQUIRED)

# Point Cloud Library (PCL)
find_package(PCL REQUIRED COMPONENTS common io visualization)

# Formatting library
find_package(fmt REQUIRED)

# ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Aggregate include directories
set(PROJECT_INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/include  # Add this line
        ${CMAKE_SOURCE_DIR}/src      # Add this line
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${spdlog_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${JSONCPP_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${rclcpp_INCLUDE_DIRS}
        ${sensor_msgs_INCLUDE_DIRS}
        ${geometry_msgs_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}
        ${image_transport_INCLUDE_DIRS}
        ${std_msgs_INCLUDE_DIRS}
        ${std_srvs_INCLUDE_DIRS}
)


# Aggregate libraries
set(PROJECT_LIBRARIES
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp
        jsoncpp
        ${PCL_LIBRARIES}
        fmt::fmt
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_TARGETS}
        ${geometry_msgs_TARGETS}
        ${cv_bridge_TARGETS}
        ${image_transport_TARGETS}
        ${std_msgs_TARGETS}
        ${std_srvs_TARGETS}
)

# Include ROS 2 specific directories
list(APPEND PROJECT_INCLUDE_DIRS

)