@echo off
setlocal enabledelayedexpansion

REM Change to the directory of this script
cd /d "%~dp0"

REM Create build directory if it doesn't exist
if exist build (
    rmdir /s /q build
)
mkdir build
cd build

REM Run cmake
cmake .. -G Ninja 
if errorlevel 1 (
    echo CMake configuration failed.
    exit /b 1
)

REM Build the project
cmake --build . --config Release
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

REM Check if executable exists
if not exist c_example.exe (
    echo c_example.exe not found!
    dir
    exit /b 1
)

REM Run the executable with arguments
echo Running c_example.exe...
.\c_example.exe .\artifacts\RuntimeLibrary_x86_64.dll .\artifacts\model.onnx .\artifacts\image.jpg "n_duplicates" "2" "n_threads_per_duplicate" "2" "runtime_log_level" "1"
