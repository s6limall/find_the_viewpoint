# cmake/PrecompiledHeaders.cmake

# List of precompiled headers to reduce compilation time
set(PRECOMPILED_HEADERS
        <iostream>
        <vector>
        <string>
        <memory>
        <numeric>
        <functional>
        <filesystem>
        <memory>
        <mutex>
        <string>
        <algorithm>
        <Eigen/Geometry>
        <Eigen/Dense>
        <pcl/PolygonMesh.h>
        <spdlog/spdlog.h>
        <yaml-cpp/yaml.h>
        <opencv2/core.hpp>
        <opencv2/opencv.hpp>
)
