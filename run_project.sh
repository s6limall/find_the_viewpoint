#!/bin/bash
# Navigate to your project directory
cd /home/kurma/Documents/University/Master/2024 SS/find_the_viewpoint/run_project.sh
# Run CMake build
cmake .
# Build the project with Make
cd ./build
make
./find_the_viewpoint
