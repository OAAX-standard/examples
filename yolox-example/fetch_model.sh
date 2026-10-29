#!/usr/bin/env bash
# Downloads an official YOLOX ONNX model (no NMS, 80 COCO classes) and checks its sha256.
#
# Usage: bash fetch_model.sh [yolox_s | yolox_tiny | yolox_nano]    (default: yolox_s)
#
# The files come from the 0.1.1rc0 release of https://github.com/Megvii-BaseDetection/YOLOX,
# an Apache-2.0 project (c) Megvii Inc. They are written to models/ and are not part of this repository.
set -euo pipefail

name="${1:-yolox_s}"
case "$name" in
  yolox_s)    sha256=c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063 ;;  # 36 MB, 640x640
  yolox_tiny) sha256=427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7 ;;  # 20 MB, 416x416
  yolox_nano) sha256=c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d ;;  # 3.7 MB, 416x416
  *) echo "Usage: $0 [yolox_s | yolox_tiny | yolox_nano]" >&2; exit 1 ;;
esac

dir="$(cd "$(dirname "$0")" && pwd)/models"
file="$dir/$name.onnx"
url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/$name.onnx"

# sha256sum on Linux, shasum on macOS
sha_of() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

mkdir -p "$dir"
if [ ! -f "$file" ] || [ "$(sha_of "$file")" != "$sha256" ]; then
  curl -fsSL --retry 3 -o "$file.part" "$url"
  if [ "$(sha_of "$file.part")" != "$sha256" ]; then
    rm -f "$file.part"
    echo "Error: sha256 of the downloaded $name.onnx does not match the pinned value" >&2
    exit 1
  fi
  mv "$file.part" "$file"
fi
echo "OK: $file (sha256 verified)"
echo "$name is (c) Megvii Inc., released under the Apache License 2.0: https://github.com/Megvii-BaseDetection/YOLOX"
