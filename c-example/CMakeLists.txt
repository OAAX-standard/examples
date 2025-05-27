cmake_minimum_required(VERSION 3.10.2)

project(c_example)

# This compiler setting is required to use dlmopen, which is a GNU extension
if(NOT WIN32)
    add_definitions(-D_GNU_SOURCE)
endif()

# 
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
set(TOOLS_C_UTILITIES_DIR ${DEPS_DIR}/tools/c-utilities)
set(JPEG_DIR ${DEPS_DIR}/jpeg/Windows)
set(TOOLS_C_UTILITIES_INCLUDE_DIR ${DEPS_DIR}/tools/c-utilities/include)

# Link C Utilities project
add_subdirectory(${TOOLS_C_UTILITIES_DIR} ${CMAKE_CURRENT_BINARY_DIR}/c_utilities)

# Include directories
add_executable(c_example src/main.c src/runtime_utils.c)

# include headers
target_include_directories(c_example PUBLIC "include" "${TOOLS_C_UTILITIES_INCLUDE_DIR}")

# link libs
if(WIN32)
    # add the JPEG library
    target_include_directories(c_example PUBLIC "${JPEG_DIR}/include")
    target_link_directories(c_example PUBLIC "${JPEG_DIR}/lib")
    # Copy jpeg62.dll to the build directory
    file(COPY "${JPEG_DIR}/bin/jpeg62.dll" DESTINATION ${CMAKE_BINARY_DIR})
    # Link the required libraries
    target_link_libraries(c_example PUBLIC jpeg c_utilities ws2_32)
else()
    target_link_libraries(c_example PUBLIC 
        -Wl,--start-group 
        pthread dl jpeg m c_utilities
        -Wl,--end-group
    )
endif()

# Copy files in the "artifacts" directory to the build directory
file(COPY artifacts DESTINATION ${CMAKE_BINARY_DIR})
