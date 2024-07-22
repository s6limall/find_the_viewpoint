#!/bin/bash

# Navigate to your project directory
cd /home/kurma/Documents/University/Master/2024\ SS/find_the_viewpoint

# Create and navigate to build directory
mkdir -p build
cd build

# Run CMake with verbose output for configuration
cmake -DCMAKE_PREFIX_PATH=/home/kurma/Documents/University/Master/2024\ SS/find_the_viewpoint/libtorch .. -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

# Build with verbose output
cmake --build . --config Release -- VERBOSE=1

# Run your project's executable
./run_find_the_view
