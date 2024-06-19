set -e

cd "$(dirname "$0")"

toolchain_path="./conversion-toolchain-latest.tar"
output_directory="../artifacts"
toolchain_container_name="cpu-toolchain-container"

output_directory=$(realpath $output_directory)

echo "Loading the toolchain image into Docker..."
docker load -i "$toolchain_path"

docker run --rm --name $toolchain_container_name -v "$output_directory:/app/run" conversion-toolchain "/app/run/model.onnx" "/app/run/"

echo "Toolchain run complete."