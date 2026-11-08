"""Compare collective-variable composition, nested updates and chain derivatives."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def reference(phase):
    return [[0.23*i+0.12*math.sin(i+phase), math.cos(1.1*i)+0.02*phase*i,
             math.sin(0.7*i)-0.03*phase*i] for i in range(8)]


def table_values(phase):
    return [(1+0.1*phase)*math.sin(0.5*i-2) for i in range(9)]


def bond(om):
    force = om.CustomBondForce("g*k*r^2")
    force.add_global_parameter("g",2)
    force.add_energy_parameter_derivative("g")
    force.add_per_bond_parameter("k")
    force.add_bond(0,1,[0.4])
    force.add_bond(2,5,[0.6])
    force.set_force_group(17)
    return force


def build(om, variant, derivatives, cache):
    cv = om.CustomCVForce("scale+g^2" if variant == 4 else "scale*(a-b)^2+0.2*f(c)")
    cv.set_force_cache_bytes(cache)
    cv.add_global_parameter("scale",1.2)
    cv.add_global_parameter("g",0.7)
    if derivatives > 0:
        cv.add_energy_parameter_derivative("scale")
    if derivatives == 2:
        cv.add_energy_parameter_derivative("g")
    if variant != 4:
        if variant == 0:
            a, b = bond(om), om.RGForce([0,2,4,6])
        elif variant == 1:
            a, b = om.OrientationRestraintForce(0.9,reference(0),[0,1,2,3,4]), bond(om)
        elif variant == 2:
            a = om.NonbondedForce()
            a.set_nonbonded_method("PME")
            a.set_cutoff_distance(1.2)
            a.set_pme_parameters(2.7,16,16,16)
            a.set_use_dispersion_correction(False)
            a.add_global_parameter("g",2)
            a.set_force_group(17)
            a.set_reciprocal_space_force_group(19)
            for i in range(8):
                a.add_particle(-0.1 if i % 2 else 0.1,0.3,0.05)
                a.add_particle_parameter_offset("g",i,-0.01 if i % 2 else 0.01,0,0)
            b = om.CustomVolumeForce("0.01*v+0.1*g")
            b.add_global_parameter("g",2)
        else:
            a = om.CustomCVForce("g*bb+0.2*rr^2")
            a.set_force_cache_bytes(cache)
            a.add_global_parameter("g",2)
            a.add_energy_parameter_derivative("g")
            a.add_collective_variable("bb",bond(om))
            a.add_collective_variable("rr",om.RGForce([0,1,2,3]))
            b = om.RGForce([0,2,4,6])
        cv.add_collective_variable("a",a)
        cv.add_collective_variable("b",b)
        cv.add_collective_variable("c",om.RMSDForce(reference(0),[0,2,4,6]))
        assert cv.get_collective_variable(0) is a
    cv.add_tabulated_function("f",om.Continuous1DFunction(table_values(0),-2,2))
    cv.set_force_group(11)
    assert cv.get_num_collective_variables() == (0 if variant == 4 else 3)
    assert cv.get_num_energy_parameter_derivatives() == derivatives
    assert cv.uses_periodic_boundary_conditions() == (variant == 2)
    return cv


def update_bond(context, force, path):
    force.set_bond_parameters(0,0,1,[0.8])
    force.set_bond_parameters(1,2,5,[0.3])
    context.update_force_parameters_at_path(path,force)


def update(cv, context, variant):
    before = context.get_state(energy=True).potential_energy
    if variant == 0:
        update_bond(context,cv.get_collective_variable(0),[0,0])
    elif variant == 1:
        force = cv.get_collective_variable(0)
        force.set_k(1.3)
        force.set_reference_positions(reference(2))
        context.update_force_parameters_at_path([0,0],force)
        update_bond(context,cv.get_collective_variable(1),[0,1])
    elif variant == 2:
        force = cv.get_collective_variable(0)
        for i in range(8):
            force.set_particle_parameters(i,-0.11 if i % 2 else 0.11,0.3,0.05)
        context.update_force_parameters_at_path([0,0],force)
    elif variant == 3:
        update_bond(context,cv.get_collective_variable(0).get_collective_variable(0),[0,0,0])
    if variant != 4:
        force = cv.get_collective_variable(2)
        force.set_reference_positions(reference(2))
        force.set_particles([1,3,5,7])
        context.update_force_parameters_at_path([0,2],force)
    before_table = context.get_state(energy=True).potential_energy
    cv.get_tabulated_function(0).set_function_parameters(table_values(2),-2,2)
    cv.set_energy_function("999")
    assert context.get_state(energy=True).potential_energy == before_table
    context.update_parameters_in_context(0,cv)
    if variant == 4:
        assert context.get_state(energy=True).potential_energy == before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm",args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(fixtures) == 360
    errors = {name: [] for name in ["energy","kinetic","force","position","velocity","parameter_derivative","collective_variable"]}
    for cache in [0,16*1024*1024]:
        key = phase = None
        for f in fixtures:
            new_key = (f["variant"],f["derivatives"])
            if key != new_key:
                key, phase = new_key, 0
                cv = build(om,*key,cache)
                system = om.System()
                for i in range(8):
                    system.add_particle(0 if i == 7 else 1+0.2*i)
                system.set_default_periodic_box_vectors((4,0,0),(0.3,4.2,0),(-0.2,0.4,4.4))
                system.add_force(cv)
                context = system.create_context(om.VerletIntegrator(0.00005))
                assert context.get_parameter("g") == 0.7
            elif phase != f["phase"]:
                phase = f["phase"]
                if phase == 1:
                    context.set_parameter("g",1.1)
                    context.set_parameter("scale",0.8)
                else:
                    update(cv,context,f["variant"])
            context.set_periodic_box_vectors((4+0.2*phase,0,0),(0.3,4.2+0.1*phase,0),(-0.2,0.4,4.4+0.15*phase))
            if f["steps"] == 0:
                context.set_positions([tuple(p) for p in f["initial_positions"]])
                context.set_velocities([tuple(p) for p in f["initial_velocities"]])
                context.set_time(0)
            else:
                context.step(f["steps"])
            state = context.get_state(forces=True,energy=True,positions=True,velocities=True,parameter_derivatives=True,groups=1<<11)
            excluded = context.get_state(forces=True,energy=True,groups=1<<10)
            assert excluded.potential_energy == 0 and all(v == 0 for p in excluded.forces for v in p)
            values = context.get_collective_variable_values(0)
            assert len(values) == len(f["values"])
            pairs = [("energy",state.potential_energy,f["energy"]),("kinetic",state.kinetic_energy,f["kinetic"])]
            pairs += [("collective_variable",a,b) for a,b in zip(values,f["values"])]
            pairs += [("parameter_derivative",state.energy_parameter_derivatives[name],f[f"derivative_{name}"]) for name in ["g","scale"]]
            for metric,attribute in [("force","forces"),("position","positions"),("velocity","velocities")]:
                pairs += [(metric,a,b) for got,want in zip(getattr(state,attribute),f[attribute]) for a,b in zip(got,want)]
            for metric,got,want in pairs:
                assert math.isfinite(got) and math.isclose(got,want,abs_tol=3e-9,rel_tol=3e-9), (key,cache,phase,f["point"],f["steps"],metric,got,want)
                errors[metric].append(abs(got-want))
    print(json.dumps({"status":"passed","reference_states":len(fixtures),"cache_modes":2,"comparisons":sum(map(len,errors.values())),"max_absolute_error":{key:max(values) for key,values in errors.items()}},indent=2))


if __name__ == "__main__":
    main()
