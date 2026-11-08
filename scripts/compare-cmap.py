"""Compare CMAP maps, shared-atom forces, periodic geometry, and short trajectories."""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def values(size, map_index, phase):
    result = []
    for j in range(size):
        for i in range(size):
            x, y = 2 * math.pi * i / size, 2 * math.pi * j / size
            result.append((1 + 0.2 * phase) * (math.sin(x) + 0.3 * math.cos(2*y) + 0.2 * math.sin(x-2*y)) + 0.1 * map_index + 0.05 * phase)
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
    assert len(fixtures) == 480, f"incomplete fixture set: {len(fixtures)}"
    errors = {"energy": [], "kinetic": [], "force": [], "position": [], "velocity": []}
    key = phase = None
    torsions = [(0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 1, 2, 3, 4)]
    for f in fixtures:
        if key != (f["size"], f["periodic"]):
            key, phase = (f["size"], f["periodic"]), 0
            force = om.CMAPTorsionForce()
            for map_index in range(2):
                size = f["size"] + map_index
                force.add_map(size, values(size, map_index, 0))
                assert force.get_map_parameters(map_index) == (size, values(size, map_index, 0))
            for index, particles in enumerate(torsions):
                force.add_torsion(index, *particles)
                assert force.get_torsion_parameters(index) == (index, *particles)
            assert force.get_num_maps() == force.get_num_torsions() == 2
            force.set_uses_periodic_boundary_conditions(f["periodic"])
            assert force.uses_periodic_boundary_conditions() == f["periodic"]
            force.set_force_group(7)
            force.set_name("CMAP fixture")
            assert force.get_force_group() == 7 and force.get_name() == "CMAP fixture"
            system = om.System()
            for i in range(8):
                system.add_particle(1 + 0.2 * i)
            system.set_default_periodic_box_vectors((4, 0, 0), (0.3, 4, 0), (-0.2, 0.4, 4))
            force_index = system.add_force(force)
            context = system.create_context(om.VerletIntegrator(0.0002))
        elif phase != f["phase"]:
            phase = f["phase"]
            previous = context.get_state(energy=True).potential_energy
            for map_index in range(2):
                size = f["size"] + map_index
                force.set_map_parameters(map_index, size, values(size, map_index, phase))
            for index, particles in enumerate(torsions):
                force.set_torsion_parameters(index, 1-index, *particles)
            assert context.get_state(energy=True).potential_energy == previous
            context.update_parameters_in_context(force_index, force)
        if f["steps"] == 0:
            context.set_positions([tuple(v) for v in f["initial_positions"]])
            context.set_velocities([tuple(v) for v in f["initial_velocities"]])
            context.set_time(0)
        else:
            context.step(f["steps"])
        state = context.get_state(forces=True, energy=True, positions=True, velocities=True, groups=1 << 7)
        excluded = context.get_state(forces=True, energy=True, groups=1 << 6)
        assert excluded.potential_energy == 0
        assert all(value == 0 for force in excluded.forces for value in force)
        pairs = [("energy", state.potential_energy, f["energy"]), ("kinetic", state.kinetic_energy, f["kinetic"])]
        for metric, attribute in [("force", "forces"), ("position", "positions"), ("velocity", "velocities")]:
            pairs += [(metric, got, want) for a, b in zip(getattr(state, attribute), f[attribute]) for got, want in zip(a, b)]
        for metric, got, want in pairs:
            assert math.isfinite(got) and math.isclose(got, want, abs_tol=2e-9, rel_tol=2e-9), (key, phase, f["point"], f["steps"], metric, got, want)
            errors[metric].append(abs(got-want))
    print(json.dumps({"status": "passed", "fixtures": len(fixtures), "comparisons": sum(map(len, errors.values())), "max_absolute_error": {key: max(error) for key, error in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
