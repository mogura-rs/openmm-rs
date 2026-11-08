"""Compare custom hydrogen bonds, tables, cutoff traversal and parameter updates."""
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
    assert len(fixtures) == 1296
    errors = {name: [] for name in ["energy", "kinetic", "force", "position", "velocity"]}
    key = phase = None
    for f in fixtures:
        new_key = (f["variant"], f["method"], f["excluded"])
        if key != new_key:
            key, phase = new_key, 0
            force = om.CustomHbondForce(f["energy_function"])
            donors = [[0,1,2],[3,4,5],[0,4,2]]
            acceptors = [[6,7,8],[9,10,11],[12,13,14]]
            if f["variant"] >= 6:
                donors = [[p[0],-1,-1] for p in donors]
                acceptors = [[p[0],-1,-1] for p in acceptors]
            force.add_global_parameter("g",1.3)
            force.add_global_parameter("x1",0.2)
            force.add_per_donor_parameter("kd")
            force.add_per_acceptor_parameter("ka")
            for i in range(3):
                force.add_donor(*donors[i],[0.7+0.1*i])
                force.add_acceptor(*acceptors[i],[0.4+0.2*i])
                assert force.get_donor_parameters(i) == (*donors[i],[0.7+0.1*i])
                assert force.get_acceptor_parameters(i) == (*acceptors[i],[0.4+0.2*i])
            if f["excluded"]:
                force.add_exclusion(0,0)
                force.add_exclusion(2,2)
            force.set_nonbonded_method(f["method"])
            force.set_cutoff_distance(1.4)
            assert force.get_num_donors() == force.get_num_acceptors() == 3
            assert force.get_nonbonded_method() == f["method"] and force.get_cutoff_distance() == 1.4
            assert force.uses_periodic_boundary_conditions() == (f["method"] == 2)
            table = om.Continuous1DFunction([math.sin(i-4) for i in range(9)], -4, 4)
            force.add_tabulated_function("f", table)
            assert force.get_energy_function() == f["energy_function"]
            force.set_force_group(9)
            system = om.System()
            for i in range(16):
                system.add_particle(1+0.2*i)
            system.set_default_periodic_box_vectors((6,0,0),(0.3,6.2,0),(-0.2,0.4,6.4))
            force_index = system.add_force(force)
            context = system.create_context(om.VerletIntegrator(0.0001))
        elif phase != f["phase"]:
            phase = f["phase"]
            if phase == 1:
                context.set_parameter("g",0.8)
                context.set_parameter("x1",0.4)
            else:
                before = context.get_state(energy=True).potential_energy
                for i in range(3):
                    force.set_donor_parameters(i,*donors[i],[1.2-0.1*i])
                    force.set_acceptor_parameters(i,*acceptors[i],[0.8-0.1*i])
                if f["excluded"]:
                    force.set_exclusion_particles(0,1,1)
                table.set_function_parameters([1.4*math.sin(i-4) for i in range(9)], -4, 4)
                assert context.get_state(energy=True).potential_energy == before
                context.update_parameters_in_context(force_index, force)
        scale = 1+0.1*phase
        context.set_periodic_box_vectors((6*scale,0,0),(0.3,6.2*scale,0),(-0.2,0.4,6.4*scale))
        if f["steps"] == 0:
            context.set_positions([tuple(v) for v in f["initial_positions"]])
            context.set_velocities([tuple(v) for v in f["initial_velocities"]])
            context.set_time(0)
        else:
            context.step(f["steps"])
        state = context.get_state(forces=True,energy=True,positions=True,velocities=True,groups=1<<9)
        excluded = context.get_state(forces=True,energy=True,groups=1<<8)
        assert excluded.potential_energy == 0 and all(v == 0 for p in excluded.forces for v in p)
        pairs = [("energy",state.potential_energy,f["energy"]),("kinetic",state.kinetic_energy,f["kinetic"])]
        for metric, attribute in [("force","forces"),("position","positions"),("velocity","velocities")]:
            pairs += [(metric,got,want) for a,b in zip(getattr(state,attribute),f[attribute]) for got,want in zip(a,b)]
        for metric,got,want in pairs:
            assert math.isfinite(got) and math.isclose(got,want,abs_tol=3e-9,rel_tol=3e-9), (key,phase,f["point"],f["steps"],metric,got,want)
            errors[metric].append(abs(got-want))
    print(json.dumps({"status":"passed","fixtures":len(fixtures),"comparisons":sum(map(len,errors.values())),"max_absolute_error":{key:max(values) for key,values in errors.items()}},indent=2))


if __name__ == "__main__":
    main()
