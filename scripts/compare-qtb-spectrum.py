import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    args = parser.parse_args()
    reference = [json.loads(row) for row in args.reference.read_text().splitlines()]
    rust = [json.loads(row) for row in args.rust.read_text().splitlines()]
    assert len(reference) == len(rust) == 58
    checks = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    timings = []
    for expected, actual in zip(reference, rust, strict=True):
        assert (expected["num_freq"], expected["temperature"], expected["friction"]) == (
            actual["n"], actual["temperature"], actual["friction"]
        )
        for key in ("theta", "deconvolved"):
            assert len(expected[key]) == len(actual[key]) == actual["n"]
            for x, y in zip(expected[key], actual[key], strict=True):
                assert math.isfinite(x) and math.isfinite(y)
                error = abs(x - y)
                assert error <= 2e-10 + 2e-12 * abs(x), (actual["n"], key, x, y)
                maximum_absolute = max(maximum_absolute, error)
                maximum_relative = max(maximum_relative, error / max(1.0, abs(x)))
                checks += 1
        if actual["n"] >= 512:
            timings.append({
                "frequencies": actual["n"],
                "reference_seconds": expected["seconds"],
                "rust_seconds": actual["seconds"],
                "speedup": expected["seconds"] / actual["seconds"],
            })
    print(json.dumps({
        "cases": len(reference), "checks": checks,
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
        "timings": timings,
    }, indent=2))


if __name__ == "__main__":
    main()
