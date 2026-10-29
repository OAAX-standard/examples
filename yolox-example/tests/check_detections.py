"""Checks the detections printed by yolox_host against the golden ones of artifacts/dog-bicycle.jpg.

Usage: yolox_host ... | uv run --no-project python tests/check_detections.py

Chips round differently (FP16, INT8, other kernels), so this compares with a tolerance:
the same classes in the same number, boxes within --pixels, scores within --score.
"""
import argparse
import re
import sys

# yolox_s on the CPU runtime (ONNX Runtime), score >= 0.30, NMS 0.45: class, score, x1, y1, x2, y2
GOLDEN = [
    ("bicycle", 0.953, 616, 210, 1228, 806),
    ("dog", 0.934, 216, 244, 521, 587),
]
LINE = re.compile(r"^\s+(\D+?)\s+(\d\.\d+)\s+\(\s*(\d+),\s*(\d+)\)\s*-\s*\(\s*(\d+),\s*(\d+)\)\s*$")


def parse(text):
    found = []
    for line in text.splitlines():
        match = LINE.match(line)
        if match:
            name, score, *box = match.groups()
            found.append((name, float(score), *map(int, box)))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixels", type=int, default=6, help="max box coordinate difference")
    parser.add_argument("--score", type=float, default=0.03, help="max score difference")
    args = parser.parse_args()

    text = sys.stdin.read()
    sys.stdout.write(text)
    found = parse(text)
    errors = []
    if len(found) != len(GOLDEN):
        errors.append(f"expected {len(GOLDEN)} detections, got {len(found)}")
    for name, score, *box in GOLDEN:
        ok = any(
            f[0] == name
            and abs(f[1] - score) <= args.score
            and all(abs(a - b) <= args.pixels for a, b in zip(f[2:], box))
            for f in found
        )
        if not ok:
            errors.append(f"missing: {name} {score} {tuple(box)}")
    if errors:
        print("\nDetections differ from the golden ones:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    print(f"\nChecked: {len(GOLDEN)} golden detections found (tolerance {args.pixels} px, {args.score} score)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
