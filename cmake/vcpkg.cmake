# cmake/vcpkg.cmake

cmake_minimum_required(VERSION 3.14...3.30)

function(setup_vcpkg)
    if (DEFINED CMAKE_TOOLCHAIN_FILE)
        set(VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}" PARENT_SCOPE)
        message(STATUS "Using pre-defined vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
        return()
    endif ()

    if (DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_ROOT $ENV{VCPKG_ROOT} PARENT_SCOPE)
    else ()
        set(VCPKG_ROOT "${CMAKE_BINARY_DIR}/vcpkg" PARENT_SCOPE)
    endif ()

    if (NOT EXISTS "${VCPKG_ROOT}/vcpkg" AND NOT EXISTS "${VCPKG_ROOT}/vcpkg.exe")
        message(STATUS "vcpkg not found, setting up in ${VCPKG_ROOT}")

        find_package(Git REQUIRED)
        execute_process(
                COMMAND "${GIT_EXECUTABLE}" clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"
                RESULT_VARIABLE GIT_RESULT
        )
        if (NOT GIT_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to clone vcpkg repository")
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
        )
        if (NOT BOOTSTRAP_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to bootstrap vcpkg")
        endif ()
    endif ()

    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")

    message(STATUS "vcpkg setup complete. VCPKG_ROOT: ${VCPKG_ROOT}")
    message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endfunction()

setup_vcpkg()
