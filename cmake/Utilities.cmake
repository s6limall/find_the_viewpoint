# cmake/Utilities.cmake

# Utility functions and settings

# Platform-specific configurations
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
elseif (UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
endif ()