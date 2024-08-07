#!/bin/bash
# Navigate to your project directory
cd /home/kurma/Documents/University/Master/2024\ SS/find_the_viewpoint/

# Run CMake build with verbose output
cmake .

# Build the project with Make, enabling verbose output

make VERBOSE=1
# Run the project executable
./find_the_viewpoint
