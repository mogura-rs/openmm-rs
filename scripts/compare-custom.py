"""Compare compiled custom energies, forces, and parameter derivatives."""

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
    assert len(fixtures) == 184, f"incomplete fixture set: {len(fixtures)}"
    errors = {"energy": [], "force": [], "parameter_derivative": []}
    for fixture in fixtures:
        kind = fixture["kind"]
        force = getattr(om, "Custom" + {"bond": "Bond", "angle": "Angle", "torsion": "Torsion", "external": "External", "nonbonded": "Nonbonded"}[kind] + "Force")(fixture["expression"])
        for name, value in [("g", 1.3), ("y", 0.7), ("r0", 0.3)]:
            force.add_global_parameter(name, value)
            if kind != "external":
                force.add_energy_parameter_derivative(name)
        if kind == "bond":
            force.add_bond(0, 1, [])
        elif kind == "angle":
            force.add_per_angle_parameter("k")
            force.add_angle(0, 1, 2, [0.8])
            force.add_angle(1, 2, 3, [1.2])
        elif kind == "torsion":
            force.add_per_torsion_parameter("k")
            force.add_torsion(0, 1, 2, 3, [0.8])
        elif kind == "external":
            force.add_per_particle_parameter("k")
            force.add_particle(0, [0.8])
            force.add_particle(2, [1.2])
        else:
            force.add_per_particle_parameter("a")
            for i in range(4):
                force.add_particle([0.8+0.2*i])
            force.add_exclusion(0, 2)
            if fixture["periodic"]:
                force.set_cutoff_distance(1.4)
        if fixture["periodic"]:
            force.set_uses_periodic_boundary_conditions(True)
        system = om.System()
        for _ in range(4):
            system.add_particle(1)
        system.set_default_periodic_box_vectors((3,0,0), (0.4,3,0), (-0.2,0.3,3))
        system.add_force(force)
        context = system.create_context(om.VerletIntegrator(0.001))
        positions = [(0,0,0), (fixture["radius"],0,0), (0.9,0.7,0.5), (1.2,0.8,1.4)]
        if fixture["periodic"]:
            positions[2] = (1.3,3.7,0.5)
            positions[3] = (1.0,1.1,4.4)
        context.set_positions(positions)
        if fixture["iteration"]:
            context.set_parameter("g",0.9)
            context.set_parameter("y",0.6)
        state = context.get_state(forces=True, energy=True, parameter_derivatives=True)
        pairs = [("energy", state.potential_energy, fixture["energy"])]
        pairs += [("force", got, want) for a,b in zip(state.forces,fixture["forces"]) for got,want in zip(a,b)]
        pairs += [("parameter_derivative", state.energy_parameter_derivatives[name], want) for name,want in fixture["derivatives"].items()]
        for metric, got, want in pairs:
            assert math.isfinite(got) and math.isclose(got,want,abs_tol=1e-10,rel_tol=1e-10), (kind, fixture["expression"], fixture["radius"], fixture["iteration"], metric, got, want)
            errors[metric].append(abs(got-want))
    print(json.dumps({"status":"passed", "fixtures":len(fixtures), "comparisons":sum(map(len,errors.values())), "max_absolute_error":{k:max(v) for k,v in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
