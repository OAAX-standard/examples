set -e

cd "$(dirname "$0")"

mkdir build || true
cd build
cmake ..
make -j
./c_example ./artifacts/libRuntimeLibrary.so ./artifacts/model.onnx ./artifacts/image.jpg
