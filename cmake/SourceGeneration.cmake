# cmake/SourceGeneration.cmake

# Add a custom command to generate sources.cmake
add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/sources.cmake
        COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_SOURCE_DIR}/cmake/generate_sources.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating sources.cmake"
        VERBATIM
)

# Add a custom target to run the above command
add_custom_target(generate_sources ALL DEPENDS ${CMAKE_BINARY_DIR}/sources.cmake)
