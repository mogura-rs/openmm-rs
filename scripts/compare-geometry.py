"""Compare geometric custom expressions, periodic forces, updates and trajectories."""
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
    assert len(fixtures) == 864
    errors = {name: [] for name in ["energy", "kinetic", "force", "position", "velocity", "parameter_derivative"]}
    key = phase = None
    for f in fixtures:
        new_key = (f["kind"], f["expression"], f["periodic"])
        if key != new_key:
            key, phase = new_key, 0
            if f["kind"] == 0:
                force = om.CustomCompoundBondForce(4, f["energy_function"])
                force.add_per_bond_parameter("k")
                force.add_bond([0,1,2,3], [0.7])
                force.add_bond([2,4,5,6], [0.9])
                force.set_uses_periodic_boundary_conditions(f["periodic"])
                table = om.Continuous1DFunction([math.sin(i-4) for i in range(9)], -4, 4)
                force.add_tabulated_function("f", table)
                assert force.get_num_particles_per_bond() == 4 and force.get_num_bonds() == 2
                assert force.get_tabulated_function_name(0) == "f"
            else:
                force = om.CustomExternalForce(f["energy_function"])
                for name in ["k", "a", "b", "c"]:
                    force.add_per_particle_parameter(name)
                force.add_particle(0, [0.7,0.1,0.2,0.3])
                force.add_particle(3, [0.9,-0.2,0.4,0.1])
            force.add_global_parameter("g", 1.3)
            if f["kind"] == 0:
                force.add_energy_parameter_derivative("g")
            assert force.uses_periodic_boundary_conditions() == (f["kind"] == 1 or f["periodic"])
            assert force.get_energy_function() == f["energy_function"]
            force.set_force_group(9)
            system = om.System()
            for i in range(8):
                system.add_particle(1+0.2*i)
            system.set_default_periodic_box_vectors((4,0,0),(0.3,4.2,0),(-0.2,0.4,4.4))
            force_index = system.add_force(force)
            context = system.create_context(om.VerletIntegrator(0.0001))
        elif phase != f["phase"]:
            phase = f["phase"]
            if phase == 1:
                context.set_parameter("g", 0.8)
            else:
                before = context.get_state(energy=True).potential_energy
                if f["kind"] == 0:
                    force.set_bond_parameters(0, [0,1,2,3], [1.2])
                    force.set_bond_parameters(1, [2,4,5,6], [0.4])
                    table.set_function_parameters([1.4*math.sin(i-4) for i in range(9)], -4, 4)
                else:
                    force.set_particle_parameters(0,0,[1.2,-0.4,0.1,0.3])
                    force.set_particle_parameters(1,3,[0.4,-0.2,0.4,-0.1])
                assert context.get_state(energy=True).potential_energy == before
                context.update_parameters_in_context(force_index, force)
        scale = 1+0.1*phase
        context.set_periodic_box_vectors((4*scale,0,0),(0.3,4.2*scale,0),(-0.2,0.4,4.4*scale))
        if f["steps"] == 0:
            context.set_positions([tuple(v) for v in f["initial_positions"]])
            context.set_velocities([tuple(v) for v in f["initial_velocities"]])
            context.set_time(0)
        else:
            context.step(f["steps"])
        state = context.get_state(forces=True,energy=True,positions=True,velocities=True,parameter_derivatives=True,groups=1<<9)
        excluded = context.get_state(forces=True,energy=True,groups=1<<8)
        assert excluded.potential_energy == 0 and all(v == 0 for p in excluded.forces for v in p)
        pairs = [("energy",state.potential_energy,f["energy"]),("kinetic",state.kinetic_energy,f["kinetic"]),("parameter_derivative",state.energy_parameter_derivatives.get("g",0),f["parameter_derivative"])]
        for metric, attribute in [("force","forces"),("position","positions"),("velocity","velocities")]:
            pairs += [(metric,got,want) for a,b in zip(getattr(state,attribute),f[attribute]) for got,want in zip(a,b)]
        for metric,got,want in pairs:
            assert math.isfinite(got) and math.isclose(got,want,abs_tol=3e-9,rel_tol=3e-9), (key,phase,f["point"],f["steps"],metric,got,want)
            errors[metric].append(abs(got-want))
    print(json.dumps({"status":"passed","fixtures":len(fixtures),"comparisons":sum(map(len,errors.values())),"max_absolute_error":{key:max(values) for key,values in errors.items()}},indent=2))


if __name__ == "__main__":
    main()
