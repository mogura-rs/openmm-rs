"""Compare virtual site positions and propagated forces with OpenMM Reference."""

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
    assert len(fixtures) == 21, "reference fixture set is incomplete"
    constructors = {
        "two": lambda: om.TwoParticleAverageSite(0, 1, 0.3, 0.7),
        "three": lambda: om.ThreeParticleAverageSite(0, 1, 2, 0.2, 0.3, 0.5),
        "out": lambda: om.OutOfPlaneSite(0, 1, 2, 0.2, 0.3, 0.4),
        "local": lambda: om.LocalCoordinatesSite([0, 1, 2], [0.2, 0.3, 0.5], [-1, 1, 0], [-1, 0, 1], (0.1, 0.2, 0.3)),
        "symmetry": lambda: om.SymmetrySite(0, (0, -1, 0), (1, 0, 0), (0, 0, 1), (0.1, 0.2, 0.3), False),
        "fractional": lambda: om.SymmetrySite(0, (0, -1, 0), (1, 0, 0), (0, 0, 1), (0.1, 0.2, 0.3), True),
        "nested": lambda: om.TwoParticleAverageSite(0, 1, 0.3, 0.7),
    }
    maximum_error = 0.0
    comparisons = 0
    for name, constructor in constructors.items():
        system = om.System()
        for mass in (2, 1, 3, 0):
            system.add_particle(mass)
        system.set_default_periodic_box_vectors((2, 0, 0), (0.3, 2.4, 0), (-0.2, 0.4, 2.8))
        system.set_virtual_site(3, constructor())
        assert system.is_virtual_site(3)
        assert system.get_virtual_site(3).get_num_particles() == constructor().get_num_particles()
        force = om.CustomExternalForce("x*x+2*y*y+3*z*z")
        force.add_particle(3, [])
        positions = [(0.1, 0.2, 0.3), (0.5, 0.1, 0.4), (0.2, 0.7, 0.8), (0, 0, 0)]
        if name == "nested":
            system.add_particle(0)
            system.set_virtual_site(4, om.TwoParticleAverageSite(3, 1, 0.6, 0.4))
            force.add_particle(4, [])
            positions.append((0, 0, 0))
        system.add_force(force)
        context = system.create_context(om.VerletIntegrator(0.001))
        context.set_positions(positions)
        context.compute_virtual_sites()
        previous = 0
        for fixture in (row for row in fixtures if row["case"] == name):
            if name == "fractional":
                context.set_positions([tuple(position) for position in fixture["positions"]])
                context.compute_virtual_sites()
            else:
                context.step(fixture["step"] - previous)
            previous = fixture["step"]
            state = context.get_state(positions=True, forces=True, energy=True)
            actual = [state.potential_energy] + [v for row in state.positions + state.forces for v in row]
            expected_forces = [list(force) for force in fixture["forces"]]
            if name == "fractional":
                expected_forces[0] = fixture["energy_gradient_force"]
            expected = [fixture["energy"]] + [v for row in fixture["positions"] + expected_forces for v in row]
            for got, want in zip(actual, expected):
                assert math.isfinite(got) and math.isclose(got, want, rel_tol=1e-8, abs_tol=1e-8), (name, previous, got, want)
                maximum_error = max(maximum_error, abs(got-want))
                comparisons += 1
    print(json.dumps({"status": "passed", "comparisons": comparisons, "max_absolute_error": maximum_error}, indent=2))


if __name__ == "__main__":
    main()
