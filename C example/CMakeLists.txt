cmake_minimum_required(VERSION 3.10.2)

project(c_example)

# Cross-compilation settings
set(CROSS_NAME "X86_64")
set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_LINKER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)

# Include directories
file(GLOB_RECURSE SRC "src/*.c")
message("SRC: ${SRC}")
add_executable(c_example ${SRC})

target_include_directories(c_example PUBLIC include)
target_link_libraries(c_example PUBLIC pthread dl)

# download this file: https://github.com/oax-standard/contributions/raw/develop/X86_64/artifacts/libRuntimeLibrary.so
# and place it in the build folder
set(RUNTIME_LIBRARY_PATH ${CMAKE_CURRENT_BINARY_DIR}/libRuntimeLibrary.so)
# check if the file exists
if(NOT EXISTS ${RUNTIME_LIBRARY_PATH})
    file(DOWNLOAD
            https://github.com/oax-standard/contributions/raw/develop/X86_64/artifacts/libRuntimeLibrary.so
            ${CMAKE_CURRENT_BINARY_DIR}/libRuntimeLibrary.so)
endif ()