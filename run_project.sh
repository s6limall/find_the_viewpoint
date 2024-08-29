#!/bin/bash

# Navigate to your project directory

cd /home/kurma/Documents/University/Master/2024\ SS/find_the_viewpoint/ || { echo "Failed to navigate to project directory"; exit 1; }
rm find_the_viewpoint
# Remove any previous builds to start fresh
#
# Run CMake with debug flags
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build the project with Make, enabling verbose output
make VERBOSE=1

# Run the project executable with GDB (GNU Debugger)
# This will help catch the invalid pointer error and give you a stack trace
echo "Running with GDB to catch errors..."
source LightGlue/venv/bin/activate
./find_the_viewpoint
