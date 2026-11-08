"""Validate group masks, reciprocal splitting and integration against OpenMM."""

import argparse
import importlib.util
import itertools
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
    assert len(fixtures) == 120, "reference fixture set is incomplete"
    maximum = 0.0
    comparisons = 0
    for (direct, method), cases in itertools.groupby(fixtures, key=lambda f: (f["direct"], f["method"])):
        system = om.System()
        system.set_default_periodic_box_vectors((3, 0, 0), (0, 3, 0), (0, 0, 3))
        force = om.NonbondedForce()
        force.set_nonbonded_method(method)
        force.set_cutoff_distance(1.2)
        force.set_ewald_error_tolerance(1e-5)
        force.set_force_group(1)
        force.set_reciprocal_space_force_group(5)
        force.set_include_direct_space(direct)
        external = om.CustomExternalForce("lambda*(x*x+2*y*y+3*z*z)")
        external.add_global_parameter("lambda", 0.2)
        external.set_force_group(31)
        for i, charge in enumerate((0.4, -0.3, 0.1, -0.2)):
            system.add_particle(2.0)
            force.add_particle(charge, 0.3+0.02*i, 0.5+0.1*i)
            external.add_particle(i, [])
        force.add_exception(0, 2, -0.02, 0.32, 0.4)
        force.add_exception(0, 1, 0, 1, 0)
        system.add_force(force)
        system.add_force(external)
        bond = om.HarmonicBondForce()
        bond.add_bond(1, 3, 0.7, 10)
        bond.set_force_group(3)
        system.add_force(bond)
        integrator = om.VerletIntegrator(0.001)
        integrator.set_integration_force_groups((1 << 1) | (1 << 3))
        context = system.create_context(integrator)
        context.set_positions([(0.1, 0.2, 0.3), (0.6, 0.4, 0.2), (2.8, 0.7, 0.4), (1.3, 1.1, 0.9)])
        context.set_velocities([(0.01, -0.02, 0.03), (0.02, 0.03, -0.01), (-0.03, 0.01, 0.02), (0.01, 0.01, -0.02)])
        previous = 0
        for fixture in cases:
            context.step(fixture["step"] - previous)
            previous = fixture["step"]
            state = context.get_state(forces=True, energy=True, positions=True, velocities=True, groups=fixture["groups"])
            actual = [state.potential_energy, state.kinetic_energy]
            expected = [fixture["energy"], fixture["kinetic"]]
            for name in ("forces", "positions", "velocities"):
                actual += [v for row in getattr(state, name) for v in row]
                expected += [v for row in fixture[name] for v in row]
            for got, want in zip(actual, expected):
                assert math.isfinite(got) and math.isclose(got, want, rel_tol=2e-5, abs_tol=1e-5), (method, direct, previous, fixture["groups"], got, want)
                maximum = max(maximum, abs(got-want))
                comparisons += 1
    print(json.dumps({"status": "passed", "comparisons": comparisons, "max_absolute_error": maximum}, indent=2))


if __name__ == "__main__":
    main()
