# Optimize the existing model using the conversion toolchain
set -e

cd "$(dirname "$0")" || exit 1
cd artifacts

# Filename of the compressed docker image
toolchain_path=conversion-toolchain-latest.tar
# Docker image name & tag
image_name=conversion-toolchain:latest
toolchain_container_name="oax-toolchain-container"
# shared folder
output_directory="$(pwd)"

echo "Loading the toolchain image into Docker..."
docker load -i "$toolchain_path"

echo "Running the toolchain on the model..."
docker stop $toolchain_container_name 2&> /dev/null || true
docker rm $toolchain_container_name 2&> /dev/null || true
docker run --name $toolchain_container_name -v "$output_directory:/app/run" $image_name "/app/run/model.onnx" "/app/run/"
