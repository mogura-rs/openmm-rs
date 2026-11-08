"""Validate nonbonded forces against the selected OpenMM source checkout."""

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
    assert len(fixtures) == 14, "reference fixture set is incomplete"
    results = []
    for fixture in fixtures:
        name = fixture["case"]
        pme = name.startswith("pme")
        periodic = pme or name.startswith("periodic")
        triclinic = "triclinic" in name
        system = om.System()
        system.set_default_periodic_box_vectors((3, 0, 0), (0.4 if triclinic else 0, 3, 0), (-0.2 if triclinic else 0, 0.3 if triclinic else 0, 3))
        force = om.NonbondedForce()
        method = "PME" if pme else "CutoffPeriodic" if periodic else "CutoffNonPeriodic" if name == "cutoff" else "NoCutoff"
        force.set_nonbonded_method(method)
        force.set_cutoff_distance(1.2)
        force.set_ewald_error_tolerance(1e-5)
        force.set_use_dispersion_correction("dispersion" in name)
        force.set_use_switching_function("switch" in name)
        force.set_switching_distance(0.8)
        force.set_exceptions_use_periodic_boundary_conditions("wrapped" in name)
        for i, charge in enumerate((0.4, -0.3, 0.1, 0.2 if "charged" in name else -0.2)):
            system.add_particle(2.0)
            force.add_particle(charge, 0.3 + 0.02*i, 0.5 + 0.1*i)
        if "exception" in name:
            force.add_exception(0, 2, -0.02, 0.32, 0.4)
            force.add_exception(0, 1, 0, 1, 0)
        system.add_force(force)
        context = system.create_context(om.VerletIntegrator(0.001))
        context.set_positions([(0.1, 0.2, 0.3), (0.6, 0.4, 0.2), (2.8, 0.7, 0.4), (1.3, 1.1, 0.9)])
        state = context.get_state(forces=True, energy=True)
        actual = [state.potential_energy] + [v for row in state.forces for v in row]
        expected = [fixture["energy"]] + [v for row in fixture["forces"] for v in row]
        for got, want in zip(actual, expected):
            assert math.isfinite(got) and math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-5), (name, got, want)
        results.append({"case": name, "energy_absolute_error": abs(actual[0]-expected[0]), "force_max_absolute_error": max(abs(a-b) for a, b in zip(actual[1:], expected[1:]))})
    print(json.dumps({"status": "passed", "comparisons": 13*len(fixtures), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
