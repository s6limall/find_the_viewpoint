# Makefile for cross-platform build using CMake and vcpkg

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# Build configuration
BUILD_TYPE ?= Release
BUILD_DIR := build

# Detect OS and set appropriate commands
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    RM := rmdir /s /q
    MKDIR := mkdir
else
    DETECTED_OS := $(shell uname -s)
    RM := rm -rf
    MKDIR := mkdir -p
endif

# Detect available tools
CMAKE := $(shell command -v cmake 2> /dev/null)
NINJA := $(shell command -v ninja 2> /dev/null)
CCACHE := $(shell command -v ccache 2> /dev/null)

# Set CMake generator
ifdef NINJA
    CMAKE_GENERATOR ?= Ninja
else
    CMAKE_GENERATOR ?= "Unix Makefiles"
endif

# Set number of build jobs
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
MAKEFLAGS += -j$(NPROC)

# CMake configuration
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

ifdef CCACHE
    CMAKE_FLAGS += -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
endif

# Phony targets
.PHONY: all clean rebuild build run help

# Default target
all: build

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	-@$(RM) "$(BUILD_DIR)" 2>/dev/null || true

# Rebuild everything
rebuild: clean all

# Build the project
build:
	@echo "Building project..."
	@$(MKDIR) "$(BUILD_DIR)"
	@cd "$(BUILD_DIR)" && $(CMAKE) -G $(CMAKE_GENERATOR) $(CMAKE_FLAGS) ..
	@$(CMAKE) --build "$(BUILD_DIR)" --config $(BUILD_TYPE)

# Run the project
run: build
	@echo "Running project..."
	@cd "$(BUILD_DIR)/bin" && ./find_the_viewpoint

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the project (default)"
	@echo "  clean   - Remove build directory"
	@echo "  rebuild - Clean and rebuild the project"
	@echo "  build   - Build the project"
	@echo "  run     - Run the built executable"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_TYPE       - Set build type (default: Release)"
	@echo "  CMAKE_GENERATOR  - Set CMake generator (default: Ninja if available, else Unix Makefiles)"
	@echo ""
	@echo "Detected configuration:"
	@echo "  OS               : $(DETECTED_OS)"
	@echo "  CMake            : $(CMAKE)"
	@echo "  Ninja            : $(NINJA)"
	@echo "  ccache           : $(CCACHE)"
	@echo "  Build jobs       : $(NPROC)"

# Include local make customization if exists
-include local.mk
