"""Compare Rust Python bindings with results from the supplied OpenMM source."""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def load_extension(path):
    spec = importlib.util.spec_from_file_location("openmm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flattened(value):
    if isinstance(value, (list, tuple)):
        for entry in value:
            yield from flattened(entry)
    else:
        yield value


def compare(om, fixtures):
    constructors = {
        "verlet": lambda: om.VerletIntegrator(0.001),
        "langevin": lambda: om.LangevinMiddleIntegrator(0.0, 2.0, 0.001),
        "langevin_zero_friction": lambda: om.LangevinMiddleIntegrator(0.0, 0.0, 0.001),
        "brownian": lambda: om.BrownianIntegrator(0.0, 2.0, 0.001),
        "verlet_constrained": lambda: om.VerletIntegrator(0.001),
        "langevin_constrained": lambda: om.LangevinMiddleIntegrator(0.0, 2.0, 0.001),
        "brownian_constrained": lambda: om.BrownianIntegrator(0.0, 2.0, 0.001),
    }
    comparisons = 0
    errors = {}
    for name, constructor in constructors.items():
        system = om.System()
        constrained = name.endswith("_constrained")
        system.add_particle(2.0)
        system.add_particle(3.0 if constrained else 0.0)
        if constrained:
            system.add_constraint(0, 1, 0.1)
        force = om.CustomExternalForce("2*x*x+3*y*y+4*z*z")
        force.add_particle(0, [])
        force.add_particle(1, [])
        system.add_force(force)
        context = system.create_context(constructor())
        if constrained:
            context.set_positions([(0, 0, 0), (0.1, 0, 0)])
            context.set_velocities([(0.3, 0.2, -0.1), (0.3, -0.1, 0.2)])
        else:
            context.set_positions([(0.2, -0.3, 0.4), (1, 2, 3)])
            context.set_velocities([(0.3, 0.2, -0.1), (4, 5, 6)])
        previous = 0
        for reference in (row for row in fixtures if row["case"] == name):
            step = reference["step"]
            context.step(step - previous)
            previous = step
            state = context.get_state(positions=True, velocities=True, forces=True, energy=True)
            for field in ("positions", "velocities", "forces", "potential_energy", "kinetic_energy"):
                actual = list(flattened(getattr(state, field)))
                expected = list(flattened(reference[field]))
                assert len(actual) == len(expected), (name, step, field)
                for got, want in zip(actual, expected):
                    assert math.isfinite(got), (name, step, field, got)
                    assert math.isclose(got, want, rel_tol=1e-8, abs_tol=1e-8), (
                        name, step, field, got, want
                    )
                    errors[field] = max(errors.get(field, 0.0), abs(got - want))
                    comparisons += 1
    return {"status": "passed", "comparisons": comparisons, "max_absolute_errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(fixtures) == 28, "reference fixture set is incomplete"
    print(json.dumps(compare(load_extension(args.extension), fixtures), indent=2))


if __name__ == "__main__":
    main()
