# File: cmake/CompilerSettings.cmake

# Include the PrecompiledHeaders.cmake file
include(${CMAKE_CURRENT_LIST_DIR}/PrecompiledHeaders.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if (MSVC)
    add_compile_options(/W4 /MP)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# Debug mode optimizations
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (MSVC)
        add_compile_options(/Zi /Od /Ob0)
    else ()
        add_compile_options(-g -Og)
    endif ()
endif ()

# Release mode optimizations
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (MSVC)
        add_compile_options(/O2)
    else ()
        add_compile_options(-O3)
    endif ()
endif ()

# LTO (Link Time Optimization)
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
if (LTO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "LTO is not supported: ${LTO_ERROR}")
endif ()

# Function to set include directories for a target
function(set_target_includes TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}
    )
endfunction()

# Enable C language if MPI is found
find_package(MPI QUIET)
if (MPI_FOUND)
    enable_language(C)
endif ()
