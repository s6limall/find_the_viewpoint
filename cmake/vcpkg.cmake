# cmake/vcpkg.cmake

cmake_minimum_required(VERSION 3.14...3.30)

function(setup_vcpkg)
    message(STATUS "Setting up vcpkg...")

    if (DEFINED CMAKE_TOOLCHAIN_FILE)
        set(VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}" PARENT_SCOPE)
        message(STATUS "Using pre-defined vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
        return()
    endif ()

    if (DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_ROOT $ENV{VCPKG_ROOT})
    else ()
        set(VCPKG_ROOT "${CMAKE_BINARY_DIR}/vcpkg")
    endif ()

    message(STATUS "VCPKG_ROOT is set to: ${VCPKG_ROOT}")

    if (NOT EXISTS "${VCPKG_ROOT}")
        message(STATUS "Creating directory: ${VCPKG_ROOT}")
        file(MAKE_DIRECTORY "${VCPKG_ROOT}")
    endif ()

    if (NOT EXISTS "${VCPKG_ROOT}/vcpkg" AND NOT EXISTS "${VCPKG_ROOT}/vcpkg.exe")
        message(STATUS "vcpkg not found, setting up in ${VCPKG_ROOT}")

        find_package(Git REQUIRED)
        execute_process(
                COMMAND "${GIT_EXECUTABLE}" clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"
                RESULT_VARIABLE GIT_RESULT
                ERROR_VARIABLE GIT_ERROR
        )
        if (NOT GIT_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to clone vcpkg repository. Error: ${GIT_ERROR}")
        endif ()

        if (WIN32)
            set(BOOTSTRAP_SCRIPT "${VCPKG_ROOT}/bootstrap-vcpkg.bat")
        else ()
            set(BOOTSTRAP_SCRIPT "${VCPKG_ROOT}/bootstrap-vcpkg.sh")
        endif ()

        execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env "${BOOTSTRAP_SCRIPT}"
                WORKING_DIRECTORY "${VCPKG_ROOT}"
                RESULT_VARIABLE BOOTSTRAP_RESULT
                ERROR_VARIABLE BOOTSTRAP_ERROR
        )
        if (NOT BOOTSTRAP_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to bootstrap vcpkg. Error: ${BOOTSTRAP_ERROR}")
        endif ()
    endif ()

    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")

    set(VCPKG_ROOT "${VCPKG_ROOT}" PARENT_SCOPE)
    message(STATUS "vcpkg setup complete. VCPKG_ROOT: ${VCPKG_ROOT}")
    message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endfunction()

setup_vcpkg()
