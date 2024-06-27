#!/bin/bash

# Create a new file with a header
printf "My current codebase:\n\n" > current_code.txt

# Find and append .cpp and .hpp files
find . \( -name "*.cpp" -o -name "*.hpp" \) ! -path "./cmake-build-debug/*" -exec sh -c '
  printf "File: $1\n" >> current_code.txt
  cat "$1" >> current_code.txt
  printf "\n" >> current_code.txt
' sh {} \;

# Append the directory structure
printf "\nMy current directory structure:\n" >> current_code.txt
tree -I 'cmake-build-debug' >> current_code.txt
