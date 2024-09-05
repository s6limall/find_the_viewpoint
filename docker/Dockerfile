# File: docker/Dockerfile

FROM ros2_humble_opencv_base:latest AS base

ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive

# Stage 1: Install system dependencies and build PCL
FROM base AS system_deps
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 \
    python3-colcon-common-extensions python3-rosdep python3-vcstool \
    python3-rosinstall-generator \
    git libeigen3-dev libspdlog-dev libyaml-cpp-dev libjsoncpp-dev libfmt-dev \
    libgl1-mesa-dev libglew-dev \
    libsm6 libxext6 libxrender-dev \
    libvtk9-dev \
    gdb valgrind \
    && rm -rf /var/lib/apt/lists/*

# Remove any existing PCL installations
RUN apt-get update && \
    apt-get remove -y libpcl-* && \
#    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Build PCL 1.14.1 from source
WORKDIR /pcl_build
RUN git clone --branch pcl-1.14.1 https://github.com/PointCloudLibrary/pcl.git && \
    cd pcl && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /pcl_build && \
    ldconfig

# Stage 2: Install ROS2 core packages
FROM system_deps AS ros_core
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Install MoveIt packages
FROM ros_core AS moveit_packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-moveit-core \
    ros-${ROS_DISTRO}-moveit-ros-planning \
    ros-${ROS_DISTRO}-moveit-ros-planning-interface \
    ros-${ROS_DISTRO}-moveit-ros-move-group \
    ros-${ROS_DISTRO}-moveit-ros-perception \
    ros-${ROS_DISTRO}-moveit-ros-warehouse \
    ros-${ROS_DISTRO}-moveit-ros-benchmarks \
    && rm -rf /var/lib/apt/lists/*

# Stage 4: Install simulation and navigation packages
FROM moveit_packages AS simulation_packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-slam-toolbox \
    && rm -rf /var/lib/apt/lists/*

# Stage 5: Install other ROS packages
FROM simulation_packages AS other_ros_packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-vision-opencv \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-common-plugins \
    ros-${ROS_DISTRO}-ament-lint \
    ros-${ROS_DISTRO}-ament-cmake-clang-format \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-tf2-tools \
    ros-${ROS_DISTRO}-tf2-ros \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Stage 6: Setup ROS workspace and XARM
FROM other_ros_packages AS workspace_setup
RUN mkdir -p /xarm_ws/src
WORKDIR /xarm_ws/src
RUN git clone -b ${ROS_DISTRO} https://github.com/xArm-Developer/xarm_ros2.git --recursive
WORKDIR /xarm_ws
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Stage 7: Build xarm workspace
FROM workspace_setup AS xarm_workspace_build
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF"

# Final stage: Setup environment
FROM xarm_workspace_build AS final
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc
RUN echo "source /xarm_ws/install/setup.bash" >> /root/.bashrc

# Create the ros2_ws for the user's project
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

RUN apt-get update && apt-get install -y \
    python3-ament-package \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for ROS and CMake
ENV AMENT_PREFIX_PATH=/opt/ros/$ROS_DISTRO:/xarm_ws/install:/ros2_ws/install
ENV CMAKE_PREFIX_PATH=/opt/ros/$ROS_DISTRO:/xarm_ws/install:/ros2_ws/install
ENV LD_LIBRARY_PATH=/opt/ros/${ROS_DISTRO}/lib:/xarm_ws/install/lib:/ros2_ws/install/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/ros/$ROS_DISTRO/lib/python3.10/site-packages:/xarm_ws/install/lib/python3.10/site-packages:$PYTHONPATH
ENV ROS_PACKAGE_PATH=/xarm_ws/src:/opt/ros/${ROS_DISTRO}/share

# Create a script to set up the development environment
RUN echo '#!/bin/bash\n\
source /opt/ros/$ROS_DISTRO/setup.bash\n\
source /xarm_ws/install/setup.bash\n\
if [ -f /ros2_ws/install/local_setup.bash ]; then\n\
    source /ros2_ws/install/local_setup.bash\n\
fi\n\
exec "$@"' > /ros2_ws/dev_env.sh && chmod +x /ros2_ws/dev_env.sh

# Create a CMake wrapper script
RUN echo '#!/bin/bash\n\
source /ros2_ws/dev_env.sh\n\
exec /usr/bin/cmake "$@"' > /usr/local/bin/cmake_wrapper.sh && \
    chmod +x /usr/local/bin/cmake_wrapper.sh

ENTRYPOINT ["/ros2_ws/dev_env.sh"]
CMD ["bash"]
