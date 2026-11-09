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
        if record.get("integrator") == "dpd":
            assert math.isfinite(record["kinetic_energy"]) and record["kinetic_energy"] > 0, record
            for got, target in zip(record["momentum"], [0.15,0.3,0.45]):
                assert math.isclose(got, target*record["particles"], rel_tol=2e-11, abs_tol=2e-10), record
        elif record.get("integrator", "").startswith(("nose_", "qtb_")):
            assert math.isfinite(record["kinetic_energy"]) and record["kinetic_energy"] > 0, record
        elif record.get("api") == "Context.step":
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
            if "force_squared_norm" in record:
                assert math.isclose(record["force_squared_norm"], records[0]["force_squared_norm"], rel_tol=1e-8, abs_tol=1e-6), records
            if "charge_squared_norm" in record:
                assert math.isclose(record["charge_squared_norm"], records[0]["charge_squared_norm"], rel_tol=1e-8, abs_tol=1e-8), records
                assert abs(record["total_charge"]-0.25) < 1e-10, record
            if "derivative_squared_norm" in record:
                assert math.isclose(record["derivative_squared_norm"], records[0]["derivative_squared_norm"], rel_tol=1e-8, abs_tol=1e-6), records
            if record.get("integrator", "").startswith("nose_") or (record.get("integrator") == "dpd" and record["temperature"] == 0):
                assert math.isclose(record["kinetic_energy"], records[0]["kinetic_energy"], rel_tol=1e-8, abs_tol=1e-7), records
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
    for particles, records in groups.items():
        if all("first_evaluation_seconds" in record for record in records):
            for key, summary in summaries[particles].items():
                rows = [record for record in records if "/".join(str(record[field]) for field in ["implementation", "api", "traversal", "cache_mode"] if field in record) == key]
                summary["median_first_evaluation_seconds"] = statistics.median(row["first_evaluation_seconds"] for row in rows)
                summary["median_setup_seconds"] = statistics.median(row["setup_seconds"] for row in rows)
    note = "Single-process CPU force/energy benchmark against OpenMM Reference; process RSS includes runtime and library overhead. Optimized OpenMM CPU/CUDA platforms and integration throughput are not measured here."
    if all(record.get("force") == "gbsa_obc" for rows in groups.values() for record in rows):
        note = "Matched f64 GBSA-OBC force/energy queries on a 0.3 nm cubic lattice, with one warmup evaluation. The box side is ceil(cuberoot(N))*0.3 nm and the cutoff is 0.8 nm. Charges, radii and scaling factors vary periodically by particle index. Energy and squared force norm are compared across implementations. Rust uses cell candidates for cutoff methods; the supplied Reference kernel scans all particle pairs. NoCutoff remains quadratic. Timings exclude construction; process RSS includes runtime, Context storage and one final force snapshot. This is an implicit-solvent force benchmark, not a complete molecular simulation or GPU comparison."
    if all(record.get("force") == "lcpo" for rows in groups.values() for record in rows):
        note = "Matched f64 LCPO force/energy evaluation on a 0.3 nm cubic lattice with one warmup and three processes per condition. Radii vary from 0.2 to 0.28 nm; coefficients are (0.7,-0.2,0.08,0.1) and surface tension is 1 kJ/mol/nm^2. Method 0 is nonperiodic and method 1 is periodic. Energy and squared force norm are checked across implementations. Rust stores contact edges and visits each triangle once; Reference builds per-particle maps and visits directed neighbor pairs. Construction is excluded from timing; peak RSS includes reusable workspaces, Context snapshots and runtime overhead. This is a surface-potential benchmark, not a complete molecular simulation or GPU comparison."
    if all(record.get("force") == "drude" for rows in groups.values() for record in rows):
        note = "Matched f64 Drude force/energy queries with N/2 Drude-parent pairs on a jittered 0.32 nm parent lattice. Model 0 is isotropic; model 1 uses two anisotropy axes from neighboring parents. Consecutive dipoles have Thole-screened interactions. Periodicity applies only to screened pairs, matching the source. Rust precomputes spring and screening coefficients and reuses one host force buffer; Reference recomputes them during force evaluation. Each condition has one warmup, 1000/64/8 queries for 512/8192/65536 particles and three processes. Energies and squared force norms are checked across implementations. Timings exclude construction; process RSS includes definitions, Context state and reusable workspaces. This is a synthetic CPU force benchmark against Reference, not a polarizable molecular simulation, optimized CPU/GPU comparison or Drude integrator benchmark."
    if all(record.get("force") == "constant_potential" for rows in groups.values() for record in rows):
        note = "Matched f64 constant-potential coordinate updates and force/energy queries on a jittered 0.32 nm lattice. One quarter of particles form two fixed electrodes with potentials +2/-2 kJ/mol/e, Gaussian widths 0.08/0.10 nm and Thomas-Fermi scales 0.005/nm. The total charge is constrained to 0.25 e and the field is (0.1,-0.05,0.08) kJ/mol/nm/e. Every timed query changes all solute coordinates smoothly, requiring a new charge solution. Solver 0 is unpreconditioned CG, solver 1 preconditioned CG, and solver 2 a cached packed-matrix solve in Rust. CG tolerance is 1e-8 kJ/mol/e, cutoff 0.8 nm, alpha 3/nm and cubic mesh size max(16,nextPowerOfTwo(ceil(6*boxSide))). Each condition has one warmup and three processes. Energy, force norm, charge norm and total charge are checked for every process. Setup through warmup and the first query are timed separately; process peak RSS includes them. This measures synthetic moving-solute force queries against Reference, not optimized CPU/GPU platforms or molecular sampling."
    if all(record.get("force") == "atm" for rows in groups.values() for record in rows):
        note = "Matched f64 ATM force/energy queries on a jittered 0.32 nm lattice, with one warmup and three processes per condition. Model 0 uses a one-particle custom quadratic potential and fixed shifts; model 1 uses periodic nonbonded interactions with a 0.8 nm cutoff, 0.6 nm switching distance and coordinate swaps between adjacent particle indices. The default soft-core expression uses lambda1=0.2, lambda2=0.6, alpha=0.1, uh=w0=0, umax=50, ubcore=1, acore=0.5 and direction=1. Both displaced states are evaluated on every query. Energies and squared force norms are checked for every process. Construction is excluded from timing; process peak RSS includes Context storage and runtime overhead. This measures synthetic composite force queries, not molecular alchemical sampling or optimized OpenMM CPU/GPU performance."
    if all(record.get("force") == "custom_many" for rows in groups.values() for record in rows):
        note = "Matched f64 three-particle force/energy queries on a jittered 0.32 nm lattice, with one warmup and three processes per condition. Model 0 uses a symmetric product of exponential pair distances; model 1 uses a central-particle angular expression. Per-particle amplitudes vary by index and lambda is 0.7. Cutoff methods use 0.65 nm. NoCutoff measures 32/64/128 particles and cutoff methods measure 128/256/512; the reference enumerates all triples even with a cutoff. Rust enumerates clique or neighbor-star candidates from reusable sparse adjacency storage. Energies and squared force norms are compared for every measured process. Timings exclude construction; process peak RSS includes retained workspaces and runtime overhead. This is a synthetic many-particle force benchmark, not complete molecular dynamics or optimized OpenMM CPU/GPU performance."
    if all(record.get("force") == "gay_berne" for rows in groups.values() for record in rows):
        note = "Matched f64 Gay-Berne force/energy evaluation on a 0.45 nm lattice with one warmup and three processes per condition. Model 0 consists of spherical particles with varying diameter, sigma and epsilon; model 1 uses axially symmetric centers and one zero-epsilon orientation particle per center, so its active-center count is half the listed particle count. All energy-axis scales are isotropic; the supplied Reference forces are conservative for these benchmark cases. Cutoff methods use a 0.9 nm cutoff and 0.7 nm switching distance. Energy and squared force norm are checked across implementations. Rust uses a scalar spherical-pair path and reusable cell candidates for cutoff methods. Construction is excluded from timing; peak RSS includes Context storage, reusable workspaces and runtime overhead. General biaxial force correctness is validated separately against energy finite differences, not the incorrect Reference torques. This measures synthetic force queries, not a complete molecular simulation or GPU performance."
    if all(record.get("force") == "custom_gb" for rows in groups.values() for record in rows):
        note = "Matched f64 CustomGB force/energy evaluation on a 0.3 nm lattice with cutoff 0.8 nm, one warmup and three processes per condition. Model 0 uses OBC expressions with two computed values and two requested global derivatives. Model 1 uses six computed values and 24 requested derivatives. All derivatives are computed during force evaluation; an extra untimed derivative request records their squared norm. Energy, force norm and derivative norm are checked across implementations. Construction is excluded from timing; process peak RSS includes Context and runtime overhead. NoCutoff is quadratic; cutoff methods use neighbor traversal. This measures synthetic expressions, not a complete force field or GPU execution."
    if all(record.get("api") == "Context.step" for rows in groups.values() for record in rows):
        note = "Single-process f64 integration with AndersenThermostat at 300 K, 37/ps, dt=0.001 ps, no potential forces and 10 warmup steps. Kinetic energy is checked against a six-standard-deviation equilibrium bound; random paths differ. Optimized CPU/GPU platforms are not measured."
    if all("integrator" in record for rows in groups.values() for record in rows):
        note = "Matched single-process f64 free-particle integration at dt=0.001 ps with 10 warmup steps. Stochastic friction is 0.7/ps. Kinetic energy is checked against the finite-time Gaussian mean and a six-standard-deviation bound. Random paths differ. Optimized CPU/GPU platforms are not measured."
    if all(record.get("integrator", "").startswith("nose_") for rows in groups.values() for record in rows):
        note = "Matched deterministic free-particle Nose-Hoover integration at 300 K, absolute collision frequency 0.7/ps, dt=0.001 ps, chain length 3, 3 MTS and 7 Yoshida-Suzuki stages, with 10 warmup steps. Pair cases use two individual atoms plus disjoint pairs with relative temperature 30 K and frequency 2/ps. Alternating atoms start at velocities (0.1,0.2,0.3) and (0.2,0.4,0.6) nm/ps, so both absolute and relative baths are exercised. Final full-particle KE is checked between implementations. This timing run is not an equilibrium or force-field test. Optimized CPU/GPU platforms are not measured."
    if all(record.get("integrator") == "dpd" for rows in groups.values() for record in rows):
        note = "Matched f64 DPD integration at dt=0.001 ps, friction 0.7/ps, cutoff 0.7 nm and 10 warmup steps. Particles start on a 0.3 nm cubic lattice in a periodic cube with side max(5,ceil(cuberoot(N)))*0.3 nm. Alternating velocities are (0.1,0.2,0.3) and (0.2,0.4,0.6) nm/ps. Potential forces are excluded. Final momentum and clock are checked; zero-temperature KE is also compared across implementations. Positive-temperature paths differ. This short timing run is not an equilibrium test. Optimized CPU/GPU platforms are not measured."
    if all(record.get("integrator", "").startswith("qtb_") for rows in groups.values() for record in rows):
        note = "Single-thread f64 free-particle QTB integration at 300 K, friction 0.7/ps, dt=0.001 ps, 33 steps per segment, cutoff 500/ps, one shared particle type and 10 warmup steps. Adaptation rate is zero for qtb_fixed and 0.01 for qtb_adaptive. Both implementations include colored-noise generation during timing; startup spectrum generation is excluded. Rust uses the approved exact FFT frequency grid, including the Nyquist bin; the supplied Reference kernel uses its original frequency grid. Random paths differ. Only finite kinetic energy, clock and step count are checked here; noise moments and adaptation are validated separately. This timing run does not establish quantum thermodynamic convergence or GPU performance."
    if all(record.get("integrator", "").startswith("custom_") for rows in groups.values() for record in rows):
        note = "Matched f64 CustomIntegrator free-particle programs at dt=0.001 ps with 10 warmup steps. Each program updates velocity and position, computes a per-DOF auxiliary variable, and performs a global sum. The vector program includes cross, vector, component extraction and dot operations. The Langevin program applies an exact Ornstein-Uhlenbeck velocity update with friction 0.7/ps. Scalar/vector KE is deterministic; Langevin KE is checked against a finite-time Gaussian six-standard-deviation bound. Rust generates random values only when referenced by an expression, whereas Reference generates unused random numbers as well. This measures program evaluation and Context overhead, not molecular force fields or GPU execution."
    if all(record.get("compound_switch_each_step") for rows in groups.values() for record in rows):
        note += " CompoundIntegrator alternates between two children every step; selection overhead is included."
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
