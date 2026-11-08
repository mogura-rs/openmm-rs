"""Validate minimization outcomes, fixed particles, constraints and force groups."""

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
    assert len(fixtures) == 6
    results = []
    for fixture in fixtures:
        name = fixture["case"]
        system = om.System()
        for i in range(3):
            system.add_particle(0 if (name == "fixed" and i == 0) or (name == "virtual" and i == 2) else 12+2*i)
        external = om.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        for parameter in ("k", "x0", "y0", "z0"):
            external.add_per_particle_parameter(parameter)
        for i, params in enumerate([(10, 0, 0, 0), (15, 0.8, 0, 0), (20, 0.2, 0.6, 0)]):
            external.add_particle(i, list(params))
        external.set_force_group(1)
        system.add_force(external)
        bonds = om.HarmonicBondForce()
        bonds.add_bond(0, 1, 0.8, 100)
        if name != "virtual":
            bonds.add_bond(1, 2, math.sqrt(0.72), 40)
        bonds.set_force_group(1)
        system.add_force(bonds)
        if name == "constrained":
            system.add_constraint(0, 1, 0.65)
        if name == "virtual":
            system.set_virtual_site(2, om.TwoParticleAverageSite(0, 1, 0.5, 0.5))
        if name == "groups":
            ignored = om.CustomExternalForce("1000*(x-3)^2")
            ignored.add_particle(0, [])
            ignored.set_force_group(4)
            system.add_force(ignored)
        integrator = om.VerletIntegrator(0.001)
        integrator.set_integration_force_groups(2)
        integrator.set_constraint_tolerance(1e-6)
        context = system.create_context(integrator)
        initial = [(0.2, -0.3, 0.1), (1.4, 0.4, -0.2), (0.6, 1.1, 0.3)]
        context.set_positions(initial)
        result = context.minimize(1e-4, 1 if name == "limited" else 0)
        assert result.termination == ("IterationLimit" if name == "limited" else "Converged"), (name, result.termination, result.iterations)
        state = context.get_state(positions=True, energy=True, groups=2)
        assert math.isclose(state.potential_energy, fixture["energy"], rel_tol=1e-8, abs_tol=1e-8), (name, state.potential_energy, fixture["energy"], result.termination)
        error = max(abs(a-b) for actual, expected in zip(state.positions, fixture["positions"]) for a, b in zip(actual, expected))
        assert error < 1e-5, (name, error, result.termination)
        if name == "fixed":
            assert max(abs(a-b) for a, b in zip(state.positions[0], initial[0])) < 2e-8
        if name == "constrained":
            assert result.max_constraint_error < 1e-6
        results.append({"case": name, "energy_absolute_error": abs(state.potential_energy-fixture["energy"]), "position_max_absolute_error": error, "iterations": result.iterations, "termination": result.termination, "constraint_error": result.max_constraint_error})
    print(json.dumps({"status": "passed", "comparisons": 60, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
