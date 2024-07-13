# cmake/vcpkg.cmake

# vcpkg.cmake

message(STATUS "Setting up vcpkg")

# Function to clone and bootstrap vcpkg if not already installed
function(install_vcpkg)
    if (NOT DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_ROOT "${CMAKE_BINARY_DIR}/vcpkg")
    else ()
        set(VCPKG_ROOT "$ENV{VCPKG_ROOT}")
    endif ()

    if (NOT EXISTS ${VCPKG_ROOT})
        message(STATUS "vcpkg not found, cloning and bootstrapping vcpkg")
        execute_process(
                COMMAND git clone https://github.com/microsoft/vcpkg.git ${VCPKG_ROOT}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if (NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to clone vcpkg repository")
        endif ()
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E env bash ${VCPKG_ROOT}/bootstrap-vcpkg.sh
                RESULT_VARIABLE BOOTSTRAP_RESULT
        )
        if (NOT BOOTSTRAP_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to bootstrap vcpkg")
        endif ()
    else ()
        message(STATUS "Using existing vcpkg installation at ${VCPKG_ROOT}")
    endif ()

    # Set VCPKG_ROOT as an environment variable
    set(ENV{VCPKG_ROOT} ${VCPKG_ROOT})
    message(STATUS "VCPKG_ROOT set to ${VCPKG_ROOT}")

    # Integrate vcpkg with CMake
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
endfunction()

# Call the function to ensure vcpkg is installed
install_vcpkg()
