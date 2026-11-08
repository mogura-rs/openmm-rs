"""Verify benchmark energy parity and summarize process peak RSS and timing."""

import json
import math
import statistics
from pathlib import Path
import sys


def main():
    directory = Path(sys.argv[1])
    groups = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        record = json.loads(path.read_text())
        memory_path = path.with_suffix(".memory.txt")
        for line in memory_path.read_text().splitlines():
            if "Maximum resident set size (kbytes):" in line:
                record["process_peak_rss_kib"] = int(line.rsplit(":", 1)[1])
            elif "maximum resident set size" in line:
                record["process_peak_rss_kib"] = int(line.split()[0]) // 1024
        assert record.get("process_peak_rss_kib", 0) > 0, memory_path
        groups.setdefault(record["particles"], []).append(record)
    for records in groups.values():
        expected = records[0]["energy"]
        for record in records:
            assert math.isclose(record["energy"], expected, rel_tol=1e-8, abs_tol=1e-6), records
    summaries = {}
    for particles, records in groups.items():
        implementations = {}
        for record in records:
            key = "/".join(str(record[field]) for field in ["implementation", "api", "traversal"] if field in record)
            implementations.setdefault(key, []).append(record)
        summaries[particles] = {
            key: {"samples": len(rows),
                  "median_evaluations_per_second": statistics.median(row["evaluations_per_second"] for row in rows),
                  "min_evaluations_per_second": min(row["evaluations_per_second"] for row in rows),
                  "max_evaluations_per_second": max(row["evaluations_per_second"] for row in rows),
                  "max_process_peak_rss_kib": max(row["process_peak_rss_kib"] for row in rows)}
            for key, rows in implementations.items()
        }
    result = {"note": "Single-process CPU force/energy benchmark against OpenMM Reference; process RSS includes runtime and library overhead. Optimized OpenMM CPU/CUDA platforms and integration throughput are not measured here.", "cases": groups, "summary": summaries}
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
