cmake_minimum_required(VERSION 3.10.2)

project(c_example)

# This compiler setting is required to use dlmopen, which is a GNU extension
add_definitions(-D_GNU_SOURCE)

# 
set(TOOLS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../tools/c-utilities)

# Cross-compilation settings
#set(CROSS_NAME "X86_64")
#set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
#set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
#set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
#set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
#set(CMAKE_LINKER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)

# Include directories
file(GLOB UTIL_FILES ${TOOLS_PATH}/src/*)
add_executable(c_example src/main.c src/runtime_utils.c src/utils.c ${UTIL_FILES})

# include headers
target_include_directories(c_example PUBLIC ${TOOLS_PATH}/include)
target_include_directories(c_example PUBLIC include)

# link libs
target_link_libraries(c_example PUBLIC pthread dl jpeg)

# Copy files in the "artifacts" directory to the build directory
file(COPY artifacts DESTINATION ${CMAKE_BINARY_DIR})
