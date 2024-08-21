# File: docker/base.Dockerfile

FROM osrf/ros:humble-desktop-full

ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal dependencies for OpenCV build
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Build and Install OpenCV with Contrib Modules
WORKDIR /opencv_build
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.10.0.zip && \
    unzip opencv.zip && unzip opencv_contrib.zip

RUN mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.10.0/modules \
          ../opencv-4.10.0 && \
    make -j$(nproc) && make install && ldconfig

# Clean up
RUN rm -rf /opencv_build
WORKDIR /