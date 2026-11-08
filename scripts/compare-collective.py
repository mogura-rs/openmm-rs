"""Compare alignment, gyration, orientation and volume forces with OpenMM."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def reference(phase):
    return [[0.2*i+0.15*math.sin(i+phase), math.cos(i*1.1)+0.02*phase*i,
             math.sin(i*0.7)-0.03*phase*i] for i in range(9)]


def selection(subset, phase):
    return [] if subset == 0 else ([0,2,4,6,8] if phase == 0 else [1,2,3,5,8])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(fixtures) == 349
    collapsed = fixtures.pop(0)
    assert collapsed == {"record":"collapsed_rg", "energy":0, "forces_finite":False}
    system = om.System()
    for _ in range(2):
        system.add_particle(1)
    system.add_force(om.RGForce())
    context = system.create_context(om.VerletIntegrator(0.001))
    context.set_positions([(1,1,1),(1,1,1)])
    state = context.get_state(energy=True, forces=True)
    assert state.potential_energy == 0 and all(v == 0 for p in state.forces for v in p)
    system = om.System()
    for _ in range(9):
        system.add_particle(1)
    system.add_force(om.RMSDForce(reference(0)))
    context = system.create_context(om.VerletIntegrator(0.001))
    rigid = fixtures[:12]
    fixtures = fixtures[12:]
    for f in rigid:
        assert f["record"] == "rigid_rmsd"
        assert 0 <= f["energy"] < 1e-7 and f["force_max"] < 1e-6, f
        context.set_positions([tuple(p) for p in f["positions"]])
        state = context.get_state(energy=True, forces=True)
        assert state.potential_energy == 0 and all(v == 0 for p in state.forces for v in p)
    errors = {name: [] for name in ["energy","kinetic","force","position","velocity"]}
    key = phase = None
    for f in fixtures:
        new_key = (f["type"], f["subset"])
        if key != new_key:
            key, phase = new_key, 0
            kind, subset = key
            if kind == 0:
                force = om.RMSDForce(reference(0), selection(subset,0))
            elif kind == 1:
                force = om.OrientationRestraintForce(1.7, reference(0), selection(subset,0))
            elif kind == 2:
                force = om.RGForce(selection(subset,0))
            else:
                force = om.CustomVolumeForce("g*(v-7)^2+0.2*ax*bx+0.3*by*cx+0.4*cy*cz")
                force.add_global_parameter("g",0.7)
                force.add_global_parameter("v",999.0)
            assert force.uses_periodic_boundary_conditions() == (kind == 3)
            if kind < 3:
                assert force.get_particles() == selection(subset,0)
            if kind < 2:
                assert force.get_reference_positions() == reference(0)
            force.set_name("collective probe")
            assert force.get_name() == "collective probe"
            force.set_force_group(12)
            system = om.System()
            for i in range(9):
                system.add_particle(0 if i == 8 else 1+0.2*i)
            force_index = system.add_force(force)
            context = system.create_context(om.VerletIntegrator(0.0002))
        elif phase != f["phase"]:
            phase = f["phase"]
            if kind < 2:
                before = context.get_state(energy=True).potential_energy
                force.set_reference_positions(reference(phase))
                force.set_particles(selection(subset,phase))
                if kind == 1:
                    force.set_k(1.7+0.4*phase)
                assert context.get_state(energy=True).potential_energy == before
                context.update_parameters_in_context(force_index,force)
            elif kind == 3:
                context.set_parameter("g",0.7+0.2*phase)
        context.set_periodic_box_vectors((2+0.1*phase,0,0),(0.3,2.2+0.1*phase,0),(-0.2,0.4,2.4+0.1*phase))
        if f["steps"] == 0:
            context.set_positions([tuple(v) for v in f["initial_positions"]])
            context.set_velocities([tuple(v) for v in f["initial_velocities"]])
            context.set_time(0)
        else:
            context.step(f["steps"])
        state = context.get_state(forces=True, energy=True, positions=True, velocities=True, groups=1<<12)
        excluded = context.get_state(forces=True, energy=True, groups=1<<11)
        assert excluded.potential_energy == 0 and all(v == 0 for p in excluded.forces for v in p)
        pairs = [("energy",state.potential_energy,f["energy"]),("kinetic",state.kinetic_energy,f["kinetic"])]
        for metric, attribute in [("force","forces"),("position","positions"),("velocity","velocities")]:
            pairs += [(metric,got,want) for a,b in zip(getattr(state,attribute),f[attribute]) for got,want in zip(a,b)]
        for metric,got,want in pairs:
            assert math.isfinite(got) and math.isclose(got,want,abs_tol=3e-9,rel_tol=3e-9), (key,phase,f["point"],f["steps"],metric,got,want)
            errors[metric].append(abs(got-want))
    print(json.dumps({"status":"passed", "fixtures":len(fixtures), "comparisons":sum(map(len,errors.values())), "collapsed_rg_exception":collapsed, "rigid_reference_max_energy":max(f["energy"] for f in rigid), "max_absolute_error":{key:max(values) for key,values in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
