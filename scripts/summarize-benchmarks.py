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
        if record.get("api") == "Context.step":
            if "expected_kinetic_energy" in record:
                expected = record["expected_kinetic_energy"]
                tolerance = max(1e-9, 1e-10*abs(expected), 6*record["kinetic_energy_stddev"])
            else:
                expected = 1.5 * record["particles"] * (1.380649e-23*6.02214076e23/1000) * 300
                tolerance = 6 * expected * math.sqrt(2 / (3 * record["particles"]))
            assert abs(record["kinetic_energy"]-expected) < tolerance, record
        if "simulation_time" in record:
            assert record["step_count"] == record["iterations"]+10,record
            assert math.isclose(record["simulation_time"],0.001*(record["iterations"]+10),rel_tol=1e-10,abs_tol=1e-10),record
        if record.get("adaptive"):
            assert record["maximum_step_size"]==0.001 and record["error_tolerance"]==1e-5,record
            assert record["clock_repair"] == (record["implementation"]=="openmm-reference"),record
        if record.get("barostat"):
            assert record["volume"] > 0 and abs(record["volume"]-record["expected_volume"]) < 6*record["volume_stddev"], record
        if "box_diagonal" in record:
            a,b,c=record["box_diagonal"]
            bx,cx,cy=record["box_tilt"]
            assert all(math.isfinite(value) for value in [a,b,c,bx,cx,cy]),record
            assert min(a,b,c)>0 and a>=2*abs(bx) and a>=2*abs(cx) and b>=2*abs(cy),record
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
            key = "/".join(str(record[field]) for field in ["implementation", "api", "traversal", "cache_mode"] if field in record)
            implementations.setdefault(key, []).append(record)
        summaries[particles] = {
            key: {"samples": len(rows),
                  "median_evaluations_per_second": statistics.median(row["evaluations_per_second"] for row in rows),
                  "min_evaluations_per_second": min(row["evaluations_per_second"] for row in rows),
                  "max_evaluations_per_second": max(row["evaluations_per_second"] for row in rows),
                  "max_process_peak_rss_kib": max(row["process_peak_rss_kib"] for row in rows)}
            for key, rows in implementations.items()
        }
    note = "Single-process CPU force/energy benchmark against OpenMM Reference; process RSS includes runtime and library overhead. Optimized OpenMM CPU/CUDA platforms and integration throughput are not measured here."
    if all(record.get("api") == "Context.step" for rows in groups.values() for record in rows):
        note = "Single-process f64 integration with AndersenThermostat at 300 K, 37/ps, dt=0.001 ps, no potential forces and 10 warmup steps. Kinetic energy is checked against a six-standard-deviation equilibrium bound; random paths differ. Optimized CPU/GPU platforms are not measured."
    if all("integrator" in record for rows in groups.values() for record in rows):
        note = "Matched single-process f64 free-particle integration at dt=0.001 ps with 10 warmup steps. Stochastic friction is 0.7/ps. Kinetic energy is checked against the finite-time Gaussian mean and a six-standard-deviation bound. Random paths differ. Optimized CPU/GPU platforms are not measured."
    if all(record.get("adaptive") for rows in groups.values() for record in rows):
        note += " Adaptive error tolerance is 1e-5 with maximum step 0.001 ps; free particles always select the cap. Reference timing includes per-step clock repair for the supplied kernel bug. The final clock and step count are checked."
    if all(record.get("barostat") for rows in groups.values() for record in rows):
        kinds = {record.get("barostat_kind", "isotropic") for rows in groups.values() for record in rows}
        assert len(kinds) == 1, kinds
        kind = next(iter(kinds))
        mode = {"isotropic": "isotropic", "anisotropic": "single-axis anisotropic", "membrane": "XY-isotropic membrane", "flexible": "fully flexible box"}[kind]
        settings = "20 bar and 300 K"
        if kind == "membrane":
            settings = "20 bar, 7 bar nm surface tension, 300 K and fixed Z=3 nm"
        note = f"Matched f64 {mode} volume moves every Verlet step at {settings}, with 100 warmup steps. Zero-strength bonds define diatomic molecules; integration force groups exclude potential forces. Initial volume is the ideal-gas mean; the final volume must remain within six equilibrium standard deviations. Random paths differ. This measures coordinate-scaling and Context overhead, not force-field throughput or equilibrium convergence. Optimized CPU/GPU platforms are not measured."
    if all(record.get("barostat_kind") == "flexible" for rows in groups.values() for record in rows):
        note += " Box shapes are unrestrained; this short timing run does not establish an equilibrium shape distribution."
    result = {"note": note, "cases": groups, "summary": summaries}
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
