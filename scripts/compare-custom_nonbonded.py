"""Compare custom pair methods, switching, groups, computed values, and tail corrections."""

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
    assert len(fixtures) == 384, f"incomplete fixture set: {len(fixtures)}"
    errors = {"energy": [], "force": [], "parameter_derivative": [], "long_range_energy": [], "long_range_derivative": []}
    case = None
    for f in fixtures:
        if case != f["case"]:
            case = f["case"]
            force = om.CustomNonbondedForce(f["expression"])
            assert force.get_nonbonded_method() == force.NoCutoff
            assert force.cutoff_distance() == 1.0
            force.set_cutoff_distance(1.25)
            assert force.get_nonbonded_method() == force.NoCutoff
            force.set_nonbonded_method(f["method"])
            force.set_use_switching_function(f["switching"])
            force.set_switching_distance(0.75)
            force.set_use_long_range_correction(f["longrange"])
            assert force.get_use_switching_function() == f["switching"]
            assert force.get_switching_distance() == 0.75
            assert force.get_use_long_range_correction() == f["longrange"]
            force.add_global_parameter("g", 1.2)
            force.add_global_parameter("u", 0.3)
            force.add_per_particle_parameter("a")
            force.add_per_particle_parameter("b")
            if f["computed"]:
                for name, expression in [("q", "a*(1+u)+lookup(b)"), ("s", "sqrt(b)*(1+0.1*g)")]:
                    index = force.add_computed_value(name, expression)
                    assert force.get_computed_value_parameters(index) == (name, expression)
                assert force.get_num_computed_values() == 2
                table = om.Continuous1DFunction([0, 0.04, 0.05, 0.08], 0, 0.3)
                force.add_tabulated_function("lookup", table)
            else:
                force.add_energy_parameter_derivative("g")
                force.add_energy_parameter_derivative("u")
            system = om.System()
            for i in range(6):
                system.add_particle(1)
                force.add_particle([0.1 * (1 + i % 3), 0.04 * (2 + i % 2)])
            force.add_exclusion(0, 1)
            force.add_exclusion(2, 4)
            groups = []
            if f["groups"] in [1, 2]:
                groups.append(({0, 1, 2, 3}, {2, 3, 4, 5}))
            if f["groups"] == 2:
                groups += [({0, 2, 4}, {1, 2, 4, 5}), ({0, 1, 2, 3}, {2, 3, 4, 5})]
            if f["groups"] == 3:
                groups.append((set(), {0, 1, 2, 3, 4, 5}))
            for group in groups:
                index = force.add_interaction_group(*group)
                assert force.get_interaction_group_parameters(index) == group
            assert force.get_num_interaction_groups() == len(groups)
            system.set_default_periodic_box_vectors((4, 0, 0), (0.3, 4, 0), (-0.2, 0.4, 4))
            force_index = system.add_force(force)
            context = system.create_context(om.VerletIntegrator(0.001))
            context.set_positions([tuple(v) for v in f["positions"]])
        if f["phase"] == 1:
            context.set_parameter("g", 0.9)
            context.set_parameter("u", 0.2)
        if f["phase"] == 2:
            for i in range(6):
                force.set_particle_parameters(i, [0.1 * (1 + i % 3) + 0.02, 0.04 * (2 + i % 2) + 0.012])
            if f["computed"]:
                table.set_function_parameters([0.005, 0.049, 0.060, 0.093], 0, 0.3)
            context.update_parameters_in_context(force_index, force)
        scale = 1.2 if f["phase"] == 3 else 1.0
        context.set_periodic_box_vectors((4*scale, 0, 0), (0.3*scale, 4*scale, 0), (-0.2*scale, 0.4*scale, 4*scale))
        state = context.get_state(forces=True, energy=True, parameter_derivatives=True)
        lrc = f["longrange"] and f["method"] == 2
        pairs = [("long_range_energy" if lrc else "energy", state.potential_energy, f["energy"])]
        pairs += [("force", got, want) for a, b in zip(state.forces, f["forces"]) for got, want in zip(a, b)]
        pairs += [("long_range_derivative" if lrc else "parameter_derivative", state.energy_parameter_derivatives[name], want) for name, want in f["derivatives"].items()]
        for metric, got, want in pairs:
            tolerance = 2e-7 if metric.startswith("long_range") else 2e-11
            assert math.isfinite(got) and math.isclose(got, want, abs_tol=tolerance, rel_tol=tolerance), (case, f["phase"], metric, got, want)
            errors[metric].append(abs(got-want))
    print(json.dumps({"status": "passed", "fixtures": len(fixtures), "comparisons": sum(map(len, errors.values())), "max_absolute_error": {key: max(value) for key, value in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
