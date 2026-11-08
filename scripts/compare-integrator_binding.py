"""Check live integrator settings, Context binding, trajectories and temperature changes."""
import argparse
import gc
import importlib.util
import json
import math
import weakref
from pathlib import Path

KB = 1.380649e-23*6.02214076e23/1000


def make_integrator(om, kind, temperature=0, friction=0.7, dt=0.0002):
    return [lambda: om.VerletIntegrator(dt),
            lambda: om.LangevinMiddleIntegrator(temperature, friction, dt),
            lambda: om.BrownianIntegrator(temperature, friction, dt),
            lambda: om.LangevinIntegrator(temperature, friction, dt)][kind]()


def make_system(om, constrained):
    system = om.System()
    for mass in [1, 2, 3, 0]:
        system.add_particle(mass)
    if constrained:
        system.add_constraint(0, 1, 1)
    bond = om.HarmonicBondForce()
    bond.add_bond(0, 2, 0.8, 2)
    bond.set_force_group(1)
    system.add_force(bond)
    external = om.CustomExternalForce("0.4*x+0.2*y^2")
    for i in range(4):
        external.add_particle(i, [])
    external.set_force_group(4)
    system.add_force(external)
    return system


def rejects(callback):
    try:
        callback()
    except RuntimeError:
        return
    raise AssertionError("invalid integrator operation was accepted")


def binding_probes(om):
    assert issubclass(om.LangevinIntegrator,om.LangevinMiddleIntegrator)
    assert type(om.LangevinIntegrator(300,1,0.001)).__name__ == "LangevinIntegrator"
    for kind in range(4):
        system = om.System()
        system.add_particle(1)
        integrator = make_integrator(om, kind)
        rejects(lambda: integrator.step(0))
        context = om.Context(system, integrator)
        assert context.get_system() is system and context.get_integrator() is integrator
        rejects(lambda: system.create_context(integrator))
        assert context.get_integrator() is integrator
        for value in [0, -0.01, math.inf, math.nan]:
            rejects(lambda: integrator.set_step_size(value))
        assert integrator.get_step_size() == 0.0002
        for value in [0, 1, -0.1, math.inf, math.nan]:
            rejects(lambda: integrator.set_constraint_tolerance(value))
        assert integrator.get_constraint_tolerance() == 1e-5
        if kind:
            for value in [-1, math.inf, math.nan]:
                rejects(lambda: integrator.set_temperature(value))
                rejects(lambda: integrator.set_friction(value))
            assert integrator.get_temperature() == 0 and integrator.get_friction() == 0.7
        context.set_positions([(0, 0, 0)])
        context.set_velocities([(1, 0, 0)])
        integrator.step(1)
        context.step(1)
        assert context.step_count == 2 and context.time == 0.0004
        reference = weakref.ref(context)
        del context
        gc.collect()
        assert reference() is None
        rejects(lambda: integrator.step(1))
        # A collected Context releases the binding without a dangling pointer.
        context = system.create_context(integrator)
        assert context.get_integrator() is integrator
        del integrator, system
        gc.collect()
        context.set_positions([(0, 0, 0)])
        context.get_integrator().step(1)
        assert context.get_system().num_particles() == 1 and context.step_count == 1
        reference = weakref.ref(context)
        retained = context.get_integrator()
        del context
        gc.collect()
        assert reference() is None
        rejects(lambda: retained.step(1))
        empty = om.System()
        rejects(lambda: empty.create_context(object()))



def seed_probes(om):
    for kind in [1, 2, 3]:
        def run(change_seed):
            system = om.System()
            for mass in [1, 2, 3, 0]:
                system.add_particle(mass)
            integrator = make_integrator(om, kind, temperature=300)
            assert integrator.getRandomNumberSeed() == 0
            integrator.setRandomNumberSeed(-17)
            assert integrator.get_random_number_seed() == -17
            context = om.Context(system, integrator)
            context.set_positions([(0, 0, 0)]*4)
            if change_seed == "before":
                integrator.set_random_number_seed(31)
            integrator.step(1)
            if change_seed == "after":
                integrator.set_random_number_seed(31)
            integrator.step(3)
            state = context.get_state(positions=True, velocities=True)
            return state.positions, state.velocities
        assert run(None) == run("before") == run("after") == run(None)
        system = om.System()
        system.add_particle(1)
        a = make_integrator(om, kind, temperature=300)
        b = make_integrator(om, kind, temperature=300)
        a.setRandomNumberSeed(31)
        b.setRandomNumberSeed(31)
        first = om.Context(system, a)
        second = om.Context(system, b)
        first.set_positions([(0, 0, 0)])
        second.set_positions([(0, 0, 0)])
        a.step(3)
        b.step(3)
        expected = first.get_state(positions=True, velocities=True)
        actual = second.get_state(positions=True, velocities=True)
        assert (actual.positions, actual.velocities) == (expected.positions, expected.velocities)
        del first
        gc.collect()
        rebound = om.Context(system, a)
        rebound.set_positions([(0, 0, 0)])
        a.step(3)
        actual = rebound.get_state(positions=True, velocities=True)
        assert (actual.positions, actual.velocities) == (expected.positions, expected.velocities)


def langevin_alias_probes(om):
    for constrained in [False,True]:
        results=[]
        for cls in [om.LangevinIntegrator,om.LangevinMiddleIntegrator]:
            system=make_system(om,constrained)
            integrator=cls(300,0.7,0.0002)
            integrator.setRandomNumberSeed(71)
            context=system.create_context(integrator)
            context.set_positions([(0,0,0),(1,0,0),(0,0.7,1.1),(0.2,0.4,0.5)])
            context.set_velocities([(0.2,0.1,-0.1),(-0.3,0.4,0.2),(0.1,-0.2,0.3),(0.7,0.8,0.9)])
            context.apply_velocity_constraints(1e-10)
            integrator.step(3)
            integrator.setTemperature(200)
            integrator.setFriction(1.3)
            integrator.setStepSize(0.0001)
            integrator.step(5)
            state=context.get_state(positions=True,velocities=True,energy=True)
            results.append((state.positions,state.velocities,state.potential_energy,state.kinetic_energy,state.time,state.step_count))
        assert results[0]==results[1]


def check_moments(moments, samples, label):
    for value, target, variance in zip(moments, [0, 1, 3], [1, 2, 96]):
        assert abs(value-target) < 6*math.sqrt(variance/samples), (label, moments, samples)


def statistics(om, rows):
    reports = []
    for kind in [1, 2, 3]:
        n = 12000
        system = om.System()
        for i in range(n):
            system.add_particle(1+i%5)
        integrator = make_integrator(om, kind, temperature=1, friction=1e6 if kind in [1, 3] else 0.7, dt=0.001)
        context = om.Context(system, integrator)
        for phase, temperature in enumerate([100, 500, 200]):
            source = next(r for r in rows if r["kind"] == kind and r["phase"] == phase)
            check_moments(source["moments"], source["samples"], ("reference", kind, phase))
            integrator.setTemperature(temperature)
            assert integrator.temperature == integrator.get_temperature() == integrator.getTemperature() == temperature
            context.set_positions([(0, 0, 0)]*n)
            context.set_velocities([(0, 0, 0)]*n)
            integrator.step(1)
            velocity = context.get_state(velocities=True).velocities
            normalized = []
            for i, v in enumerate(velocity):
                variance = KB*temperature/(1+i%5)
                if kind == 2:
                    variance *= 2/(0.7*0.001)
                normalized.extend(x/math.sqrt(variance) for x in v)
            moments = [sum(z**power for z in normalized)/(3*n) for power in [1, 2, 4]]
            check_moments(moments, 3*n, ("rust", kind, phase))
            reports.append({"kind": kind, "temperature": temperature, "samples": 3*n,
                            "reference_moments": source["moments"], "rust_moments": moments})
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    fixtures = [row for row in rows if not row.get("statistic")]
    assert len(fixtures) == 96 and len(rows) == 105
    errors = {key: [] for key in ["energy", "kinetic", "time", "force", "position", "velocity"]}
    previous = None
    for f in fixtures:
        key = f["kind"], f["constrained"]
        if previous != key:
            kind, constrained = key
            system = make_system(om, constrained)
            integrator = make_integrator(om, kind)
            context = om.Context(system, integrator)
            assert context.get_system() is system and context.get_integrator() is integrator
            context.set_positions([(0,0,0), (1,0,0), (0,0.7,1.1), (0.2,0.4,0.5)])
            context.set_velocities([(0.2,0.1,-0.1), (-0.3,0.4,0.2), (0.1,-0.2,0.3), (0.7,0.8,0.9)])
            context.apply_velocity_constraints(1e-10)
            context.set_time(2.7)
            context.set_step_count(17)
            previous = key
        if f["steps"] == 0:
            phase = f["phase"]
            dt = [0.0002, 0.00035, 0.0001, 0.0005][phase]
            groups = [-1, 2, 0, 18][phase]
            friction = [0.7, 1.4, 2.1, 0.3][phase]
            tolerance = [1e-7, 1e-8, 1e-9, 1e-10][phase]
            context.get_integrator().setStepSize(dt)
            integrator.setIntegrationForceGroups(groups)
            integrator.setConstraintTolerance(tolerance)
            assert integrator.step_size == integrator.get_step_size() == integrator.getStepSize() == dt
            assert integrator.get_integration_force_groups() == integrator.getIntegrationForceGroups() == groups
            assert integrator.get_constraint_tolerance() == integrator.getConstraintTolerance() == tolerance
            if kind:
                integrator.setFriction(friction)
                assert integrator.friction == integrator.get_friction() == integrator.getFriction() == friction
            completed = 0
        if f["steps"] == 1:
            context.step(f["steps"]-completed)
        else:
            integrator.step(f["steps"]-completed)
        completed = f["steps"]
        state = context.get_state(energy=True, forces=True, positions=True, velocities=True)
        assert state.step_count == f["step_count"]
        pairs = [("energy", state.potential_energy, f["energy"]),
                 ("kinetic", state.kinetic_energy, f["kinetic"]), ("time", state.time, f["time"])]
        for metric, name in [("force", "forces"), ("position", "positions"), ("velocity", "velocities")]:
            pairs += [(metric, got, want) for a,b in zip(getattr(state, name), f[name]) for got,want in zip(a,b)]
        for metric, got, want in pairs:
            assert math.isfinite(got) and math.isclose(got, want, abs_tol=2e-8, rel_tol=2e-8), (
                key, f["phase"], f["steps"], metric, got, want)
            errors[metric].append(abs(got-want))
    binding_probes(om)
    langevin_alias_probes(om)
    seed_probes(om)
    reports = statistics(om, [row for row in rows if row.get("statistic")])
    print(json.dumps({"status": "passed", "fixtures": len(fixtures), "binding_probes": "passed",
                      "comparisons": sum(map(len, errors.values())), "statistics": reports,
                      "max_absolute_error": {key: max(values) for key,values in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
