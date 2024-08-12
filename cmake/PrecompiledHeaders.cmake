# Organize precompiled headers by category
set(STD_HEADERS
        <iostream>
        <vector>
        <string>
        <memory>
        <numeric>
        <functional>
        <filesystem>
        <algorithm>
        <mutex>
)

set(EIGEN_HEADERS
        <Eigen/Core>
        <Eigen/Geometry>
        <Eigen/Dense>
)

set(PCL_HEADERS
        <pcl/point_types.h>
        <pcl/point_cloud.h>
        <pcl/common/common.h>
        <pcl/PolygonMesh.h>
)

set(OPENCV_HEADERS
        <opencv2/core.hpp>
        <opencv2/imgproc.hpp>
        <opencv2/highgui.hpp>
)

set(UTILITY_HEADERS
        <spdlog/spdlog.h>
        <yaml-cpp/yaml.h>
)

# Function to add precompiled headers to a target
function(add_precompiled_headers TARGET_NAME)
    if (NOT ${CMAKE_VERSION} VERSION_LESS "3.16")
        target_precompile_headers(${TARGET_NAME}
                PRIVATE
                ${STD_HEADERS}
                ${EIGEN_HEADERS}
                ${PCL_HEADERS}
                ${OPENCV_HEADERS}
                ${UTILITY_HEADERS}
        )
    else ()
        message(WARNING "CMake version does not support precompiled headers. Skipping.")
    endif ()
endfunction()
