"""Compare molecular connectivity and periodic wrapping with OpenMM."""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(fixtures) == 3
    maximum = 0.0
    for fixture in fixtures:
        system = om.System()
        for i in range(12):
            system.add_particle(0 if i == 3 else 1)
        skew = fixture["skew"]
        system.set_default_periodic_box_vectors((3, 0, 0), (skew, 3, 0), (-skew, skew, 3))
        bond = om.HarmonicBondForce()
        bond.add_bond(0, 1, 0.5, 1)
        system.add_force(bond)
        angle = om.HarmonicAngleForce()
        angle.add_angle(1, 2, 4, 1, 1)
        system.add_force(angle)
        system.add_constraint(5, 6, 0.5)
        torsion = om.PeriodicTorsionForce()
        torsion.add_torsion((6, 7, 8, 9), 2, 0, 1)
        system.add_force(torsion)
        custom = om.CustomBondForce("r")
        custom.add_bond(10, 11, [])
        system.add_force(custom)
        nonbonded = om.NonbondedForce()
        for _ in range(12):
            nonbonded.add_particle(0, 1, 0)
        nonbonded.add_exception(4, 5, 0, 1, 0)
        system.add_force(nonbonded)
        system.set_virtual_site(3, om.TwoParticleAverageSite(1, 2, 0.3, 0.7))
        context = system.create_context(om.VerletIntegrator(0.001))
        context.set_positions([(-4.2+0.17*i, 6.4+0.23*i, -1.5+0.11*i) for i in range(12)])
        context.compute_virtual_sites()
        assert context.get_molecules() == fixture["molecules"], (context.get_molecules(), fixture["molecules"])
        for name, wrapped in [("wrapped", True), ("raw", False)]:
            positions = context.get_state(positions=True, enforce_periodic_box=wrapped).positions
            for actual, expected in zip(positions, fixture[name]):
                for a, e in zip(actual, expected):
                    assert math.isclose(a, e, rel_tol=1e-10, abs_tol=1e-10), (name, a, e)
                    maximum = max(maximum, abs(a-e))
    print(json.dumps({"status": "passed", "comparisons": 216, "max_absolute_error": maximum}, indent=2))


if __name__ == "__main__":
    main()
