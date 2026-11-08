"""Compare CompoundIntegrator routing, live settings, ownership and nested execution."""
import argparse
import gc
import importlib.util
import json
import math
import weakref
from pathlib import Path


def reject(callback):
    try:
        callback()
    except (RuntimeError, OverflowError):
        return
    raise AssertionError("invalid compound operation was accepted")


def make(om, kind):
    value = [lambda: om.VerletIntegrator(0.0002),
             lambda: om.LangevinMiddleIntegrator(0, 0.7, 0.0003),
             lambda: om.BrownianIntegrator(0, 0.9, 0.0002),
             lambda: om.VariableVerletIntegrator(1e-5),
             lambda: om.VariableLangevinIntegrator(0, 0.8, 1e-5),
             lambda: om.LangevinIntegrator(0, 0.6, 0.0004)][kind]()
    if kind in [3, 4]:
        value.setMaximumStepSize(0.0003)
    value.setConstraintTolerance(1e-10)
    return value


def create(om, row):
    system = om.System()
    for mass in [1, 2, 3, 0]:
        system.add_particle(mass)
    if row["constrained"]:
        system.add_constraint(0, 1, 1)
    if row["virtual_site"]:
        system.set_virtual_site(3, om.TwoParticleAverageSite(0, 2, 0.3, 0.7))
    bond = om.HarmonicBondForce()
    bond.add_bond(0, 2, 0.8, 2)
    bond.set_force_group(1)
    system.add_force(bond)
    force = om.CustomExternalForce("0.4*x+0.2*y^2")
    for i in range(4):
        force.add_particle(i, [])
    force.set_force_group(4)
    system.add_force(force)
    root = om.CompoundIntegrator()
    branches = [om.CompoundIntegrator(), om.CompoundIntegrator()] if row["nested"] else []
    for branch in branches:
        root.addIntegrator(branch)
    leaves = [make(om, (i+row["rotation"]) % 6) for i in range(6)]
    for i, leaf in enumerate(leaves):
        (branches[i//3] if branches else root).addIntegrator(leaf)
    context = om.Context(system, root)
    context.set_positions([(0, 0, 0), (1, 0, 0), (0, 0.7, 1.1), (0.2, 0.4, 0.5)])
    context.compute_virtual_sites()
    context.set_velocities([(0.2, 0.1, -0.1), (-0.3, 0.4, 0.2), (0.1, -0.2, 0.3), (0.7, 0.8, 0.9)])
    context.apply_velocity_constraints(1e-10)
    context.set_time(2.7)
    context.set_step_count(17)
    return context, root, branches, leaves


def ownership_probes(om):
    system = om.System()
    system.add_particle(1)
    empty = om.CompoundIntegrator()
    assert empty.getNumIntegrators() == 0 and empty.getCurrentIntegrator() == 0
    for action in [lambda: om.Context(system, empty), lambda: empty.getStepSize(),
                   lambda: empty.getConstraintTolerance(), lambda: empty.getIntegrationForceGroups(),
                   lambda: empty.setCurrentIntegrator(0), lambda: empty.step(0),
                   lambda: empty.addIntegrator(empty)]:
        reject(action)
    child = om.VerletIntegrator(0.001)
    root = om.CompoundIntegrator()
    branch = om.CompoundIntegrator()
    assert root.addIntegrator(branch) == 0
    reject(lambda: branch.addIntegrator(root))
    reject(lambda: om.Context(system, root))
    assert branch.addIntegrator(child) == 0
    assert root.getIntegrator(0) is branch and branch.getIntegrator(0) is child
    assert root.get_integrator(0) is branch
    reject(lambda: root.addIntegrator(child))
    reject(lambda: om.CompoundIntegrator().addIntegrator(child))
    reject(lambda: om.Context(system, child))
    reject(lambda: om.Context(system, branch))
    reject(lambda: root.getIntegrator(9))
    reject(lambda: branch.setCurrentIntegrator(2))
    context = om.Context(system, root)
    assert context.get_integrator() is root
    reject(lambda: om.Context(system, root))
    reject(lambda: root.addIntegrator(om.VerletIntegrator(0.001)))
    reject(lambda: branch.addIntegrator(om.VerletIntegrator(0.001)))
    for value in [0, -1, math.nan, math.inf]:
        reject(lambda: root.setStepSize(value))
    for value in [0, 1, -1, math.nan, math.inf]:
        reject(lambda: root.setConstraintTolerance(value))
    context.set_positions([(0, 0, 0)])
    context.set_velocities([(1, 0, 0)])
    root.step(1)
    branch.step(1)
    child.step(1)
    assert context.step_count == 3 and context.time == 0.003
    ref = weakref.ref(context)
    del context
    gc.collect()
    assert ref() is None
    reject(lambda: child.step(1))
    reject(lambda: root.addIntegrator(om.VerletIntegrator(0.001)))
    rebound = om.Context(system, root)
    rebound.set_positions([(0, 0, 0)])
    child.step(1)
    assert rebound.step_count == 1
    del rebound, root, branch
    gc.collect()
    # Retained children become reusable after their owner is collected.
    independent = om.Context(system, child)
    independent.set_positions([(0, 0, 0)])
    child.step(1)
    assert independent.step_count == 1
    del independent
    gc.collect()
    variable = om.VariableVerletIntegrator(1e-5)
    variable.setMaximumStepSize(0.001)
    root = om.CompoundIntegrator()
    branch = om.CompoundIntegrator()
    branch.addIntegrator(variable)
    root.addIntegrator(child)
    root.addIntegrator(branch)
    root.setCurrentIntegrator(1)
    root.setStepSize(0)
    assert variable.getStepSize() == root.getStepSize() == 0
    root.setCurrentIntegrator(0)
    context = om.Context(system, root)
    context.set_positions([(0, 0, 0)])
    context.set_velocities([(1, 0, 0)])
    root.step(1)
    variable.stepTo(0.00371)
    root.step(1)
    assert context.step_count == 5 and math.isclose(context.time, 0.00471)
    assert math.isclose(context.get_state(positions=True).positions[0][0], 0.00471)


def update_probe(om):
    system = om.System()
    system.add_particle(1)
    system.add_force(om.AndersenThermostat(0, 1e9))
    root = om.CompoundIntegrator()
    variable = om.VariableVerletIntegrator(1e-5)
    variable.setMaximumStepSize(0.001)
    fixed = om.VerletIntegrator(0.001)
    root.addIntegrator(variable)
    root.addIntegrator(fixed)
    context = om.Context(system, root)
    context.set_positions([(0, 0, 0)])
    context.set_velocities([(1, 2, 3)])
    fixed.step(1)
    assert context.get_state(velocities=True).velocities == [(1, 2, 3)]
    root.setCurrentIntegrator(1)
    variable.step(1)
    assert context.get_state(velocities=True).velocities == [(0, 0, 0)]


def seed_probe(om):
    def run(interleave):
        system = om.System()
        for _ in range(8):
            system.add_particle(1)
        root = om.CompoundIntegrator()
        children = [om.LangevinMiddleIntegrator(300, 0.7, 0.001) for _ in range(2)]
        for i, child in enumerate(children):
            child.setRandomNumberSeed(71+i)
            root.addIntegrator(child)
        context = om.Context(system, root)
        values = []
        for i in range(6):
            if interleave:
                children[1].step(3)
            context.set_positions([(0, 0, 0)]*8)
            context.set_velocities([(0, 0, 0)]*8)
            children[0].step(1)
            values.append(context.get_state(velocities=True).velocities)
        return values
    assert run(False) == run(True) == run(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(rows) == 1152
    errors = {key: [] for key in ["energy", "kinetic", "time", "dt", "tolerance", "position", "velocity", "force"]}
    previous = None
    masks = [-1, 2, 16, 18]
    for row in rows:
        key = tuple(row[k] for k in ["rotation", "nested", "constrained", "virtual_site"])
        if key != previous:
            context, root, branches, leaves = create(om, row)
            previous = key
        event = row["event"]
        active = event % 6
        called = active if event % 2 == 0 else (active+2) % 6
        if row["sample"] == 0:
            if branches:
                root.setCurrentIntegrator(active//3)
                branches[active//3].setCurrentIntegrator(active % 3)
            else:
                root.setCurrentIntegrator(active)
            root.setStepSize(0.0001+0.00003*(event % 5))
            root.setConstraintTolerance(1e-9/(1+event % 3))
            if event % 4 == 0:
                root.setIntegrationForceGroups(masks[(event//4) % 4])
            leaves[called].setIntegrationForceGroups(masks[(event+1) % 4])
            if event == 10:
                context.set_velocities_to_temperature(0)
        else:
            (root if event % 2 == 0 else leaves[called]).step(1)
        assert root.getCurrentIntegrator() == row["current"]
        assert [leaf.getIntegrationForceGroups() for leaf in leaves] == row["groups"]
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True)
        assert state.step_count == row["step_count"]
        pairs = [("energy", state.potential_energy, row["energy"]),
                 ("kinetic", state.kinetic_energy, row["kinetic"]), ("time", state.time, row["time"])]
        pairs += [("dt", leaf.getStepSize(), target) for leaf, target in zip(leaves, row["dt"])]
        pairs += [("tolerance", leaf.getConstraintTolerance(), target) for leaf, target in zip(leaves, row["tolerance"])]
        for metric, field in [("position", "positions"), ("velocity", "velocities"), ("force", "forces")]:
            pairs += [(metric, got, want) for a, b in zip(getattr(state, field), row[field]) for got, want in zip(a, b)]
        for metric, got, want in pairs:
            assert math.isfinite(got) and math.isclose(got, want, rel_tol=2e-8, abs_tol=2e-8), (key, event, row["sample"], metric, got, want)
            errors[metric].append(abs(got-want))
    ownership_probes(om)
    update_probe(om)
    seed_probe(om)
    print(json.dumps({"status": "passed", "fixtures": len(rows), "comparisons": sum(map(len, errors.values())),
                      "ownership_probes": "passed", "context_update_probe": "passed", "seed_probe": "passed",
                      "max_absolute_error": {key: max(values) for key, values in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
