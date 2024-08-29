# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/kurma/Documents/University/Master/2024 SS/find_the_viewpoint"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/kurma/Documents/University/Master/2024 SS/find_the_viewpoint"

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start "/home/kurma/Documents/University/Master/2024 SS/find_the_viewpoint/CMakeFiles" "/home/kurma/Documents/University/Master/2024 SS/find_the_viewpoint//CMakeFiles/progress.marks"
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start "/home/kurma/Documents/University/Master/2024 SS/find_the_viewpoint/CMakeFiles" 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -P "/home/kurma/Documents/University/Master/2024 SS/find_the_viewpoint/CMakeFiles/VerifyGlobs.cmake"
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named find_the_viewpoint

# Build rule for target.
find_the_viewpoint: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 find_the_viewpoint
.PHONY : find_the_viewpoint

# fast build rule for target.
find_the_viewpoint/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/build
.PHONY : find_the_viewpoint/fast

src/ApplyLightGlueWrapper.o: src/ApplyLightGlueWrapper.cpp.o
.PHONY : src/ApplyLightGlueWrapper.o

# target to build an object file
src/ApplyLightGlueWrapper.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/ApplyLightGlueWrapper.cpp.o
.PHONY : src/ApplyLightGlueWrapper.cpp.o

src/ApplyLightGlueWrapper.i: src/ApplyLightGlueWrapper.cpp.i
.PHONY : src/ApplyLightGlueWrapper.i

# target to preprocess a source file
src/ApplyLightGlueWrapper.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/ApplyLightGlueWrapper.cpp.i
.PHONY : src/ApplyLightGlueWrapper.cpp.i

src/ApplyLightGlueWrapper.s: src/ApplyLightGlueWrapper.cpp.s
.PHONY : src/ApplyLightGlueWrapper.s

# target to generate assembly for a file
src/ApplyLightGlueWrapper.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/ApplyLightGlueWrapper.cpp.s
.PHONY : src/ApplyLightGlueWrapper.cpp.s

src/config/config.o: src/config/config.cpp.o
.PHONY : src/config/config.o

# target to build an object file
src/config/config.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/config/config.cpp.o
.PHONY : src/config/config.cpp.o

src/config/config.i: src/config/config.cpp.i
.PHONY : src/config/config.i

# target to preprocess a source file
src/config/config.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/config/config.cpp.i
.PHONY : src/config/config.cpp.i

src/config/config.s: src/config/config.cpp.s
.PHONY : src/config/config.s

# target to generate assembly for a file
src/config/config.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/config/config.cpp.s
.PHONY : src/config/config.cpp.s

src/image.o: src/image.cpp.o
.PHONY : src/image.o

# target to build an object file
src/image.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/image.cpp.o
.PHONY : src/image.cpp.o

src/image.i: src/image.cpp.i
.PHONY : src/image.i

# target to preprocess a source file
src/image.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/image.cpp.i
.PHONY : src/image.cpp.i

src/image.s: src/image.cpp.s
.PHONY : src/image.s

# target to generate assembly for a file
src/image.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/image.cpp.s
.PHONY : src/image.cpp.s

src/main.o: src/main.cpp.o
.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i
.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s
.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

src/path_and_vis.o: src/path_and_vis.cpp.o
.PHONY : src/path_and_vis.o

# target to build an object file
src/path_and_vis.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/path_and_vis.cpp.o
.PHONY : src/path_and_vis.cpp.o

src/path_and_vis.i: src/path_and_vis.cpp.i
.PHONY : src/path_and_vis.i

# target to preprocess a source file
src/path_and_vis.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/path_and_vis.cpp.i
.PHONY : src/path_and_vis.cpp.i

src/path_and_vis.s: src/path_and_vis.cpp.s
.PHONY : src/path_and_vis.s

# target to generate assembly for a file
src/path_and_vis.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/path_and_vis.cpp.s
.PHONY : src/path_and_vis.cpp.s

src/processing/dfs.o: src/processing/dfs.cpp.o
.PHONY : src/processing/dfs.o

# target to build an object file
src/processing/dfs.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/processing/dfs.cpp.o
.PHONY : src/processing/dfs.cpp.o

src/processing/dfs.i: src/processing/dfs.cpp.i
.PHONY : src/processing/dfs.i

# target to preprocess a source file
src/processing/dfs.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/processing/dfs.cpp.i
.PHONY : src/processing/dfs.cpp.i

src/processing/dfs.s: src/processing/dfs.cpp.s
.PHONY : src/processing/dfs.s

# target to generate assembly for a file
src/processing/dfs.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/processing/dfs.cpp.s
.PHONY : src/processing/dfs.cpp.s

src/sift.o: src/sift.cpp.o
.PHONY : src/sift.o

# target to build an object file
src/sift.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/sift.cpp.o
.PHONY : src/sift.cpp.o

src/sift.i: src/sift.cpp.i
.PHONY : src/sift.i

# target to preprocess a source file
src/sift.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/sift.cpp.i
.PHONY : src/sift.cpp.i

src/sift.s: src/sift.cpp.s
.PHONY : src/sift.s

# target to generate assembly for a file
src/sift.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/sift.cpp.s
.PHONY : src/sift.cpp.s

src/task1.o: src/task1.cpp.o
.PHONY : src/task1.o

# target to build an object file
src/task1.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/task1.cpp.o
.PHONY : src/task1.cpp.o

src/task1.i: src/task1.cpp.i
.PHONY : src/task1.i

# target to preprocess a source file
src/task1.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/task1.cpp.i
.PHONY : src/task1.cpp.i

src/task1.s: src/task1.cpp.s
.PHONY : src/task1.s

# target to generate assembly for a file
src/task1.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/task1.cpp.s
.PHONY : src/task1.cpp.s

src/task2.o: src/task2.cpp.o
.PHONY : src/task2.o

# target to build an object file
src/task2.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/task2.cpp.o
.PHONY : src/task2.cpp.o

src/task2.i: src/task2.cpp.i
.PHONY : src/task2.i

# target to preprocess a source file
src/task2.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/task2.cpp.i
.PHONY : src/task2.cpp.i

src/task2.s: src/task2.cpp.s
.PHONY : src/task2.s

# target to generate assembly for a file
src/task2.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/find_the_viewpoint.dir/build.make CMakeFiles/find_the_viewpoint.dir/src/task2.cpp.s
.PHONY : src/task2.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... find_the_viewpoint"
	@echo "... src/ApplyLightGlueWrapper.o"
	@echo "... src/ApplyLightGlueWrapper.i"
	@echo "... src/ApplyLightGlueWrapper.s"
	@echo "... src/config/config.o"
	@echo "... src/config/config.i"
	@echo "... src/config/config.s"
	@echo "... src/image.o"
	@echo "... src/image.i"
	@echo "... src/image.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
	@echo "... src/path_and_vis.o"
	@echo "... src/path_and_vis.i"
	@echo "... src/path_and_vis.s"
	@echo "... src/processing/dfs.o"
	@echo "... src/processing/dfs.i"
	@echo "... src/processing/dfs.s"
	@echo "... src/sift.o"
	@echo "... src/sift.i"
	@echo "... src/sift.s"
	@echo "... src/task1.o"
	@echo "... src/task1.i"
	@echo "... src/task1.s"
	@echo "... src/task2.o"
	@echo "... src/task2.i"
	@echo "... src/task2.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -P "/home/kurma/Documents/University/Master/2024 SS/find_the_viewpoint/CMakeFiles/VerifyGlobs.cmake"
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

