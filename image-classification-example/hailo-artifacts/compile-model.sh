set -e

cd "$(dirname "$0")"

toolchain_path="./onnx-to-hailo-latest.tar"
hailo_deps_directory="./hailo_deps"
artifacts_directory="../artifacts"
toolchain_container_name="onnx-to-hailo-container"

artifacts_directory=$(realpath $artifacts_directory)
hailo_deps_directory=$(realpath $hailo_deps_directory)

echo "Creating a zip file containing the ONNX model, the JSON and the calibration images"
cp $artifacts_directory/model.onnx ./model.onnx
zip -r $artifacts_directory/input.zip images info.json model.onnx

echo "Loading the toolchain image into Docker..."
docker load -i "$toolchain_path"

docker run -v $hailo_deps_directory:/app/hailo-deps \
    -v $artifacts_directory:/app/input \
    -v $artifacts_directory:/app/output \
    onnx-to-hailo:latest /app/input/input.zip /app/output


echo "Toolchain run complete."