"""Compare live Python force definitions and isolated Context snapshots."""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path


def make_system(om, kind):
    system = om.System()
    for mass in range(1, 5):
        system.add_particle(mass)
    extra = {}
    if kind == 0:
        root = om.HarmonicBondForce()
        root.add_bond(0, 1, 0.3, 1)
    elif kind == 1:
        root = om.CustomNonbondedForce("a*s1*s2*f(r)")
        root.add_per_particle_parameter("s")
        root.add_global_parameter("a", 0.4)
        for i in range(4):
            root.add_particle([0.2*(i+1)])
        extra["table"] = om.Continuous1DFunction([0, 0.2, 0.1, 0.3], 0, 3)
        root.add_tabulated_function("f", extra["table"])
    elif kind == 2:
        root = om.CustomCVForce("scale*q^2")
        nested = om.CustomCVForce("c*v")
        child = om.CustomBondForce("k*(r-b)^2")
        child.add_global_parameter("b", 0.1)
        child.add_per_bond_parameter("k")
        child.add_bond(0, 1, [0.2])
        nested.add_collective_variable("v", child)
        nested.add_global_parameter("c", 0.3)
        root.add_collective_variable("q", nested)
        root.add_global_parameter("scale", 0.4)
        extra.update(nested=nested, child=child)
    else:
        root = om.AndersenThermostat(100, 2)
        external = om.CustomExternalForce("0.2*x^2")
        for i in range(4):
            external.add_particle(i, [])
        system.add_force(external)
    index = system.add_force(root)
    assert system.get_force(index) is root
    assert system.get_forces()[index] is root
    assert system.num_forces() == (2 if kind == 3 else 1)
    return system, root, extra


def mutate(system, root, extra, kind, phase):
    root.set_force_group(phase+1)
    root.set_name("edited force")
    system.set_particle_mass(0, 2+phase)
    side = 4+0.2*phase
    system.set_default_periodic_box_vectors((side, 0, 0), (0, side, 0), (0, 0, side))
    if kind == 0:
        root.set_bond_parameters(0, phase, 3, 0.4+0.1*phase, 3+phase)
        root.set_uses_periodic_boundary_conditions(phase != 1)
    elif kind == 1:
        root.set_global_parameter_default_value(0, 0.7+phase)
        root.set_particle_parameters(1, [1.5+phase])
        root.set_nonbonded_method(root.CutoffPeriodic if phase == 1 else root.NoCutoff)
        root.set_cutoff_distance(1.6)
        extra["table"].set_function_parameters([0.1+phase, 0.4, 0.8, 0.3+phase], 0, 3)
    elif kind == 2:
        root.set_global_parameter_default_value(0, 0.7+phase)
        extra["nested"].set_global_parameter_default_value(0, 1.1+0.2*phase)
        extra["child"].set_global_parameter_default_value(0, 0.4+0.1*phase)
        extra["child"].set_bond_parameters(0, phase, 3, [0.5+phase])
        extra["child"].set_uses_periodic_boundary_conditions(phase != 1)
    else:
        root.set_default_temperature(200+50*phase)
        root.set_default_collision_frequency(7+phase)


def rejects(callback, error=RuntimeError):
    try:
        callback()
    except error:
        return
    raise AssertionError("invalid ownership operation was accepted")


def ownership_probes(om):
    system, second = om.System(), om.System()
    force = om.HarmonicBondForce()
    assert system.add_force(force) == 0
    rejects(lambda: system.add_force(force))
    rejects(lambda: second.add_force(force))
    cv = om.CustomCVForce("x")
    rejects(lambda: cv.add_collective_variable("x", force))
    assert system.num_forces() == 1 and second.num_forces() == 0
    assert cv.get_num_collective_variables() == 0
    rejects(lambda: system.get_force(1), IndexError)
    rejects(lambda: system.remove_force(1), IndexError)
    rejects(lambda: system.add_force(object()))
    assert system.num_forces() == 1
    system.remove_force(0)
    assert not system.get_forces()
    assert cv.add_collective_variable("x", force) == 0
    rejects(lambda: second.add_force(force))
    rejects(lambda: cv.add_collective_variable("y", force))
    assert cv.get_num_collective_variables() == 1
    assert second.add_force(cv) == 0
    rejects(lambda: system.add_force(cv))
    second.remove_force(0)
    del cv
    gc.collect()
    assert system.add_force(force) == 0
    del system
    gc.collect()
    assert second.add_force(force) == 0
    del second
    gc.collect()
    # A retained force remains safe after removing or deleting its owner.
    force.set_name("retained")
    assert force.get_name() == "retained"
    a, b = om.CustomCVForce("x"), om.CustomCVForce("x")
    rejects(lambda: a.add_collective_variable("x", a))
    assert a.get_num_collective_variables() == 0
    a.add_collective_variable("x", b)
    rejects(lambda: b.add_collective_variable("x", a))
    assert b.get_num_collective_variables() == 0
    owner = om.System()
    owner.add_force(a)
    full = om.CustomCVForce("0")
    for i in range(32):
        full.add_collective_variable(f"v{i}", om.HarmonicBondForce())
    unused = om.HarmonicBondForce()
    rejects(lambda: full.add_collective_variable("overflow", unused))
    assert full.get_num_collective_variables() == 32
    owner.add_force(unused)
    # Owner references keep temporary forces and nested variables alive.
    system = om.System()
    for _ in range(2):
        system.add_particle(1)
    for k in [2, 5]:
        force = om.HarmonicBondForce()
        force.add_bond(0, 1, 0.5, k)
        system.add_force(force)
    del force
    gc.collect()
    system.remove_force(0)
    context = system.create_context(om.VerletIntegrator(0.001))
    context.set_positions([(0, 0, 0), (1, 0, 0)])
    assert context.get_state(energy=True).potential_energy == 0.625
    system.remove_force(0)
    assert context.get_state(energy=True).potential_energy == 0.625
    empty = system.create_context(om.VerletIntegrator(0.001))
    empty.set_positions([(0, 0, 0), (1, 0, 0)])
    assert empty.get_state(energy=True).potential_energy == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(fixtures) == 288
    errors = {name: [] for name in ["energy", "kinetic", "force", "parameter"]}
    previous = None
    for f in fixtures:
        key = f["kind"], f["phase"]
        if key != previous:
            kind, phase = key
            if phase == 0:
                system, root, extra = make_system(om, kind)
                contexts = []
            mutate(system, root, extra, kind, phase)
            contexts.append(system.create_context(om.VerletIntegrator(0.0002)))
            previous = key
        assert system.uses_periodic_boundary_conditions() == f["system_pbc"]
        context = contexts[f["context"]]
        point = f["point"]
        context.set_positions([(0.1,0.2,0.3), (0.7+point*0.03,0.4,0.5),
                               (1.4,0.9+point*0.02,0.2), (3.5,0.3,0.7+point*0.01)])
        context.set_velocities([(0.2,-0.1,0.3), (-0.3,0.2,0.1), (0.1,0.3,-0.2), (0.4,-0.2,0.1)])
        state = context.get_state(energy=True, forces=True, parameters=True, groups=f["groups"])
        assert state.parameters.keys() == f["parameters"].keys()
        pairs = [("energy", state.potential_energy, f["energy"]),
                 ("kinetic", state.kinetic_energy, f["kinetic"])]
        pairs += [("force", got, want) for a,b in zip(state.forces, f["forces"]) for got,want in zip(a,b)]
        pairs += [("parameter", state.parameters[name], value) for name,value in f["parameters"].items()]
        for metric, got, want in pairs:
            assert math.isfinite(got) and math.isclose(got, want, abs_tol=2e-10, rel_tol=2e-10), (
                key, f["context"], f["point"], f["groups"], metric, got, want)
            errors[metric].append(abs(got-want))
    ownership_probes(om)
    print(json.dumps({"status": "passed", "fixtures": len(fixtures),
                      "ownership_probes": "passed", "comparisons": sum(map(len, errors.values())),
                      "max_absolute_error": {key: max(values) for key,values in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
