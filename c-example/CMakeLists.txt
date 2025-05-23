cmake_minimum_required(VERSION 3.10.2)

project(c_example)

# This compiler setting is required to use dlmopen, which is a GNU extension
add_definitions(-D_GNU_SOURCE)

# 
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
set(TOOLS_C_UTILITIES_DIR ${DEPS_DIR}/tools/c-utilities)
set(TOOLS_C_UTILITIES_INCLUDE_DIR ${DEPS_DIR}/tools/c-utilities/include)

# Link C Utilities project
add_subdirectory(${TOOLS_C_UTILITIES_DIR} ${CMAKE_CURRENT_BINARY_DIR}/c_utilities)

# Include directories
add_executable(c_example src/main.c src/runtime_utils.c)

# include headers
  
target_include_directories(c_example PUBLIC "include" "${TOOLS_C_UTILITIES_INCLUDE_DIR}")

# link libs
target_link_libraries(c_example PUBLIC 
    -Wl,--start-group 
    pthread dl jpeg m c_utilities 
    -Wl,--end-group
)

# Copy files in the "artifacts" directory to the build directory
file(COPY artifacts DESTINATION ${CMAKE_BINARY_DIR})
