# File: cmake/SourceManagement.cmake

# Function to add sources to the project
function(add_project_sources TARGET_NAME)
    # Define source directories
    set(SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Initialize lists for sources and headers
    set(PROJECT_SOURCES "")
    set(PROJECT_HEADERS "")

    # Recursive helper function to add sources
    function(add_sources_recursive PARENT_DIR)
        # Get all files in the current directory
        file(GLOB CURRENT_SOURCES "${PARENT_DIR}/*.cpp" "${PARENT_DIR}/*.c")
        file(GLOB CURRENT_HEADERS "${PARENT_DIR}/*.hpp" "${PARENT_DIR}/*.h")

        # Add sources to the main list
        list(APPEND PROJECT_SOURCES ${CURRENT_SOURCES})
        list(APPEND PROJECT_HEADERS ${CURRENT_HEADERS})

        # Recursively add sources from subdirectories
        file(GLOB SUBDIRS "${PARENT_DIR}/*")
        foreach (SUBDIR ${SUBDIRS})
            if (IS_DIRECTORY "${SUBDIR}")
                add_sources_recursive("${SUBDIR}")
            endif ()
        endforeach ()

        # Update parent scope variables
        set(PROJECT_SOURCES ${PROJECT_SOURCES} PARENT_SCOPE)
        set(PROJECT_HEADERS ${PROJECT_HEADERS} PARENT_SCOPE)
    endfunction()

    # Call the recursive function for each source directory
    foreach (DIR ${SOURCE_DIRS})
        add_sources_recursive(${DIR})
    endforeach ()

    # Remove duplicates
    list(REMOVE_DUPLICATES PROJECT_SOURCES)
    list(REMOVE_DUPLICATES PROJECT_HEADERS)

    # Add sources to the target
    target_sources(${TARGET_NAME} PRIVATE ${PROJECT_SOURCES} ${PROJECT_HEADERS})

    # Set source groups for better organization in IDEs
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${PROJECT_SOURCES} ${PROJECT_HEADERS})

    # Print the list of sources and headers (optional, for debugging)
    #[[message(STATUS "Project sources: ${PROJECT_SOURCES}")
    message(STATUS "Project headers: ${PROJECT_HEADERS}")]]
endfunction()
