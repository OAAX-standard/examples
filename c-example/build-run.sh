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

./c_example "$runtime_library" "./artifacts/model.onnx" "./artifacts/image.jpg" \
    "n_duplicates" "2" \
    "n_threads_per_duplicate" "2" \
    "runtime_log_level" "1"
