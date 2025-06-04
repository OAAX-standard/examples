set -e

cd "$(dirname "$0")"

mkdir build || true
cd build
cmake ..
make -j

# Choose the appropriate library based on the architecture
if [[ "$(uname -m)" == "x86_64" ]]; then
    runtime_library="./artifacts/libRuntimeLibrary_x86_64.so"
elif [[ "$(uname -m)" == "aarch64" ]]; then
    runtime_library="./artifacts/libRuntimeLibrary_aarch64.so"
else
    echo "Unsupported architecture: $(uname -m)"
    exit 1
fi

# Usage: ./c_example
#   <library_path> <model_path> <image_path> <number_of_inferences> // Program parameters
#   <input_height> <input_width> <nchw> <mean> <std>                // Model parameters
#   <key1> <value1> <key2> <value2> ... <keyN> <valueN>             // Runtime parameters
./c_example "$runtime_library" "./artifacts/model.onnx" "./artifacts/image.jpg" 10 \
    240 320 1 127 128 \
    "n_duplicates" "2" \
    "n_threads_per_duplicate" "2" \
    "runtime_log_level" "2" \
    "queue_capacity" "10"
