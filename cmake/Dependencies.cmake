# File: cmake/Dependencies.cmake

include(FindPackageHandleStandardArgs)

# Ensure vcpkg is being used
if (NOT DEFINED VCPKG_ROOT)
    message(FATAL_ERROR "VCPKG_ROOT is not defined. Make sure vcpkg.cmake is included before this file.")
endif ()

# Option to prefer system packages
option(PREFER_SYSTEM_PACKAGES "Prefer system packages over vcpkg if available" ON)

# Define the list of required packages with their arguments
set(REQUIRED_PACKAGES
        "OpenCV"
        "Eigen3 NO_MODULE"
        "spdlog"
        "yaml-cpp CONFIG"
        "jsoncpp CONFIG"
        "PCL COMPONENTS common io visualization"
        "fmt CONFIG"
)

# Initialize project-wide variables
set(PROJECT_INCLUDE_DIRS "")
set(PROJECT_LIBRARIES "")
set(PROJECT_DEFINITIONS "")

# Helper function to find and configure packages
function(find_and_configure_package PACKAGE_NAME)
    if (PREFER_SYSTEM_PACKAGES)
        find_package(${PACKAGE_NAME} ${ARGN} QUIET)
        if (${PACKAGE_NAME}_FOUND)
            message(STATUS "Found system ${PACKAGE_NAME} ${${PACKAGE_NAME}_VERSION}")
        else ()
            find_package(${PACKAGE_NAME} ${ARGN} REQUIRED)
            message(STATUS "Found vcpkg ${PACKAGE_NAME} ${${PACKAGE_NAME}_VERSION}")
        endif ()
    else ()
        find_package(${PACKAGE_NAME} ${ARGN} REQUIRED)
        message(STATUS "Found vcpkg ${PACKAGE_NAME} ${${PACKAGE_NAME}_VERSION}")
    endif ()

    if (${PACKAGE_NAME}_INCLUDE_DIRS)
        list(APPEND PROJECT_INCLUDE_DIRS ${${PACKAGE_NAME}_INCLUDE_DIRS})
    endif ()
    if (${PACKAGE_NAME}_LIBRARIES)
        list(APPEND PROJECT_LIBRARIES ${${PACKAGE_NAME}_LIBRARIES})
    endif ()
    if (${PACKAGE_NAME}_DEFINITIONS)
        list(APPEND PROJECT_DEFINITIONS ${${PACKAGE_NAME}_DEFINITIONS})
    endif ()

    set(PROJECT_INCLUDE_DIRS ${PROJECT_INCLUDE_DIRS} PARENT_SCOPE)
    set(PROJECT_LIBRARIES ${PROJECT_LIBRARIES} PARENT_SCOPE)
    set(PROJECT_DEFINITIONS ${PROJECT_DEFINITIONS} PARENT_SCOPE)
endfunction()

# Find required packages
foreach (PACKAGE_SPEC ${REQUIRED_PACKAGES})
    string(REPLACE " " ";" PACKAGE_LIST ${PACKAGE_SPEC})
    list(GET PACKAGE_LIST 0 PACKAGE_NAME)
    list(REMOVE_AT PACKAGE_LIST 0)
    find_and_configure_package(${PACKAGE_NAME} ${PACKAGE_LIST})
endforeach ()

# Special handling for Eigen3 (it uses a different naming convention)
if (TARGET Eigen3::Eigen)
    list(APPEND PROJECT_LIBRARIES Eigen3::Eigen)
endif ()

# Special handling for fmt and yaml-cpp
if (TARGET fmt::fmt)
    list(APPEND PROJECT_LIBRARIES fmt::fmt)
endif ()

if (TARGET yaml-cpp::yaml-cpp)
    list(APPEND PROJECT_LIBRARIES yaml-cpp::yaml-cpp)
endif ()

# Add project's include directory
list(APPEND PROJECT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Remove duplicates
list(REMOVE_DUPLICATES PROJECT_INCLUDE_DIRS)
list(REMOVE_DUPLICATES PROJECT_LIBRARIES)
list(REMOVE_DUPLICATES PROJECT_DEFINITIONS)

# Print found package information
foreach (PACKAGE_SPEC ${REQUIRED_PACKAGES})
    string(REPLACE " " ";" PACKAGE_LIST ${PACKAGE_SPEC})
    list(GET PACKAGE_LIST 0 PACKAGE_NAME)
    if (${PACKAGE_NAME}_FOUND)
        message(STATUS "${PACKAGE_NAME} version: ${${PACKAGE_NAME}_VERSION}")
    endif ()
endforeach ()
