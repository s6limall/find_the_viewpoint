# cmake/CompilerSettings.cmake

# Compiler and optimization settings

# Integrate ccache for faster builds (if available)
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    message(STATUS "ccache found and integrated.")
else ()
    message(STATUS "ccache not found. Building without ccache.")
endif ()

# Enable C language if MPI is found
find_package(MPI QUIET)
if (MPI_FOUND)
    enable_language(C)
endif ()

# Enable compiler warnings for better code quality
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    #    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
endif ()

# Set build types to ensure predictable builds
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING "Available build types" FORCE)

# Compiler optimizations for all build types
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -flto -march=native)
    add_link_options(-O3 -flto -march=native)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/GL /Ox /O2)
    add_link_options(/LTCG)
endif ()

# Set preferred generator to Ninja if available for faster builds
if (NOT CMAKE_GENERATOR STREQUAL "Ninja")
    set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "Ninja generator is preferred for faster builds")
endif ()
