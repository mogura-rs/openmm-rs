"""Compare tabulated energies, chain-rule forces, derivatives, and live updates."""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def parameters(fixture):
    dimension = fixture["dimension"]
    sizes = fixture["shape"][:dimension] if dimension > 1 else []
    result = [*sizes, fixture["values"]]
    if not fixture["discrete"]:
        result += [value for pair in fixture["bounds"][:dimension] for value in pair]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    endpoint_fixture = fixtures.pop()
    assert endpoint_fixture["endpoint_roundoff"]
    assert len(fixtures) == 1134, f"incomplete fixture set: {len(fixtures)}"
    errors = {"energy": [], "force": [], "parameter_derivative": []}
    case = phase = None
    radius = 0.731
    for fixture in fixtures:
        if case != fixture["case"]:
            case, phase = fixture["case"], 0
            force = om.CustomNonbondedForce(fixture["expression"])
            for name in ("g", "u", "v", "w"):
                force.add_global_parameter(name, 0)
                force.add_energy_parameter_derivative(name)
            kind = "Discrete" if fixture["discrete"] else "Continuous"
            constructor = getattr(om, f"{kind}{fixture['dimension']}DFunction")
            kwargs = {} if fixture["discrete"] else {"periodic": fixture["periodic"]}
            table = constructor(*parameters(fixture), **kwargs)
            force.add_tabulated_function("f", table)
            assert force.get_num_tabulated_functions() == 1
            assert force.get_tabulated_function_name(0) == "f"
            assert table.get_periodic() == fixture["periodic"]
            for _ in range(2):
                force.add_particle([])
            system = om.System()
            for _ in range(2):
                system.add_particle(1)
            force_index = system.add_force(force)
            context = system.create_context(om.VerletIntegrator(0.001))
            context.set_positions([(0, 0, 0), (radius, 0, 0)])
            context.set_parameter("g", 1.2)
        elif phase != fixture["phase"]:
            phase = fixture["phase"]
            previous_energy = context.get_state(energy=True).potential_energy
            count = table.get_update_count()
            force.get_tabulated_function(0).set_function_parameters(*parameters(fixture))
            assert table.get_update_count() == count + 1
            assert context.get_state(energy=True).potential_energy == previous_energy
            context.update_parameters_in_context(force_index, force)
        expected = parameters(fixture)
        actual = table.get_function_parameters()
        actual = [actual] if fixture["discrete"] and fixture["dimension"] == 1 else list(actual)
        assert actual == expected
        p = fixture["coordinates"]
        context.set_parameter("u", p[0] - math.sin(radius))
        context.set_parameter("v", p[1] - 0.2 * radius)
        context.set_parameter("w", p[2] + 0.3 * radius)
        state = context.get_state(forces=True, energy=True, parameter_derivatives=True)
        pairs = [("energy", state.potential_energy, fixture["energy"])]
        pairs += [("force", got, want) for a, b in zip(state.forces, fixture["forces"]) for got, want in zip(a, b)]
        pairs += [("parameter_derivative", state.energy_parameter_derivatives[name], want) for name, want in fixture["derivatives"].items()]
        for metric, got, want in pairs:
            assert math.isfinite(got) and math.isclose(got, want, abs_tol=2e-10, rel_tol=2e-10), (case, phase, fixture["point"], metric, got, want)
            errors[metric].append(abs(got-want))
    force = om.CustomNonbondedForce("f(r)")
    force.add_tabulated_function("f", om.Continuous1DFunction([1, 2, 3, 4, 5, 6], -1.1, 2.2))
    system = om.System()
    for _ in range(2):
        system.add_particle(1)
        force.add_particle([])
    system.add_force(force)
    context = system.create_context(om.VerletIntegrator(0.001))
    context.set_positions([(0, 0, 0), (2.2, 0, 0)])
    state = context.get_state(forces=True, energy=True)
    assert math.isclose(state.potential_energy, 6, abs_tol=1e-14)
    assert all(math.isfinite(value) for force in state.forces for value in force)
    if endpoint_fixture["source_rejected"]:
        assert "outside the range" in endpoint_fixture["error"]
    else:
        assert math.isclose(state.potential_energy, endpoint_fixture["energy"], abs_tol=1e-14)
    print(json.dumps({"status": "passed", "fixtures": len(fixtures), "comparisons": sum(map(len, errors.values())), "max_absolute_error": {k: max(v) for k, v in errors.items()}, "endpoint_roundoff": {"rust_energy": state.potential_energy, "source_rejected": endpoint_fixture["source_rejected"]}}, indent=2))


if __name__ == "__main__":
    main()
