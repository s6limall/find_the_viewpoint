# Stage 1: Base image with ROS2 Humble
FROM osrf/ros:humble-desktop AS ros_base

ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive

# Stage 2: Install system dependencies
FROM ros_base AS system_deps
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    python3-rosinstall-generator \
    python3-ament-package \
    git \
    libeigen3-dev \
    libpcl-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libjsoncpp-dev \
    libfmt-dev \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Build and Install OpenCV with Contrib Modules
FROM system_deps AS opencv_build
WORKDIR /opencv_build

# Download OpenCV and OpenCV Contrib
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.4.zip && \
    unzip opencv.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.4.zip && \
    unzip opencv_contrib.zip

# Build OpenCV with Contrib modules
RUN mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.4/modules \
          ../opencv-4.5.4 && \
    make -j$(nproc) && make install && ldconfig

# Stage 4: Install ROS2 core packages
FROM opencv_build AS ros_core
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Stage 5: Install MoveIt packages
FROM ros_core AS moveit_packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-moveit-core \
    ros-${ROS_DISTRO}-moveit-ros-planning \
    ros-${ROS_DISTRO}-moveit-ros-planning-interface \
    && rm -rf /var/lib/apt/lists/*

# Stage 6: Install simulation packages
FROM moveit_packages AS simulation_packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Stage 7: Install other ROS packages
FROM simulation_packages AS other_ros_packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-xacro \
    && rm -rf /var/lib/apt/lists/*

# Stage 8: Install Python packages
FROM other_ros_packages AS python_packages
RUN pip3 install -U \
    rosdep \
    rosinstall_generator \
    vcstool \
    ament_package

# Stage 9: Setup ROS workspace and XARM
FROM python_packages AS workspace_setup
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws
RUN git clone -b ${ROS_DISTRO} https://github.com/xArm-Developer/xarm_ros2.git src/xarm_ros2 --recursive
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Stage 10: Build workspace
FROM workspace_setup AS workspace_build
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Final stage: Setup environment
FROM workspace_build AS final
RUN echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Copy the built workspace to the final image
COPY --from=workspace_build /ros2_ws /ros2_ws

# Set environment variables for ROS and CMake
ENV AMENT_PREFIX_PATH=/opt/ros/$ROS_DISTRO:/ros2_ws/install
ENV CMAKE_PREFIX_PATH=/opt/ros/$ROS_DISTRO:/ros2_ws/install

# Create a script to set up the development environment
RUN echo '#!/bin/bash\n\
source /opt/ros/$ROS_DISTRO/setup.bash\n\
source /ros2_ws/install/setup.bash\n\
exec "$@"' > /ros2_ws/dev_env.sh && chmod +x /ros2_ws/dev_env.sh

# Create a CMake wrapper script
RUN echo '#!/bin/bash\n\
source /opt/ros/$ROS_DISTRO/setup.bash\n\
source /ros2_ws/install/setup.bash\n\
export PYTHONPATH=$PYTHONPATH:/opt/ros/$ROS_DISTRO/lib/python3.10/site-packages\n\
export AMENT_PREFIX_PATH=/opt/ros/$ROS_DISTRO:$AMENT_PREFIX_PATH\n\
export CMAKE_PREFIX_PATH=/opt/ros/$ROS_DISTRO:$CMAKE_PREFIX_PATH\n\
exec /usr/bin/cmake "$@"' > /usr/local/bin/cmake_wrapper.sh && \
    chmod +x /usr/local/bin/cmake_wrapper.sh

ENTRYPOINT ["/ros2_ws/dev_env.sh"]
CMD ["bash"]
