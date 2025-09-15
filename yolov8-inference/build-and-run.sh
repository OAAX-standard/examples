#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Go to the directory of the script
cd "$(dirname "$0")"

###################### Define the model and config paths here ######################
MODEL_PATH="" # Path to your ONNX model
CONFIG_PATH="" # Path to your model config JSON. Please choose one from model-configs folder
RUNTIME_LIBRARY_PATH="" # Path to the OAAX runtime library
#####################################################################################

# Ensure all required paths are provided
if [ -z "$MODEL_PATH" ] || [ -z "$CONFIG_PATH" ] || [ -z "$RUNTIME_LIBRARY_PATH" ]; then
    echo "Error: Please set MODEL_PATH, CONFIG_PATH, and RUNTIME_LIBRARY_PATH in the script."
    exit 1
fi

# Create build directory if it doesn't exist
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

# Go to the build directory
cd "$BUILD_DIR"
# Clean previous builds
rm -rf *
# Build the project
cmake ..
make -j
# Run the executable
runtime_dir=$(dirname "$RUNTIME_LIBRARY_PATH")
export LD_LIBRARY_PATH=$runtime_dir:$LD_LIBRARY_PATH
./yolov8_inference --library "$RUNTIME_LIBRARY_PATH" --model "$MODEL_PATH" --config "$CONFIG_PATH" --input "../artifacts/image.jpg"