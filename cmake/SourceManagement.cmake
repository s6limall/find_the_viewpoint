set(SOURCES_CMAKE_FILE ${CMAKE_BINARY_DIR}/sources.cmake)

# Generate sources.cmake on every build by creating a phony target
add_custom_target(generate_sources ALL
        COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_SOURCE_DIR}/cmake/generate_sources.sh ${CMAKE_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating sources.cmake"
        VERBATIM
)

# Function to add sources to a target
function(add_project_sources target)
    # Make sure the target depends on generate_sources
    add_dependencies(${target} generate_sources)

    # Include the generated sources.cmake if it exists
    if (EXISTS ${SOURCES_CMAKE_FILE})
        include(${SOURCES_CMAKE_FILE})
        # Add sources and headers to the target
        target_sources(${target} PRIVATE ${PROJECT_SOURCES} ${PROJECT_HEADERS})
    endif ()

    # Add include directories
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endfunction()

# Function to be called after project creation to ensure sources are generated
function(ensure_sources_generated)
    if (NOT EXISTS ${SOURCES_CMAKE_FILE})
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_SOURCE_DIR}/cmake/generate_sources.sh ${CMAKE_BINARY_DIR}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                RESULT_VARIABLE GENERATE_RESULT
        )
        if (NOT GENERATE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to generate sources.cmake")
        endif ()
    endif ()
endfunction()
