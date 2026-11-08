"""Compare center-of-mass removal schedules and trajectories with OpenMM."""
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
    assert len(fixtures) == 576
    errors = {name: [] for name in ["energy", "kinetic", "time", "force", "position", "velocity"]}
    key = None
    for f in fixtures:
        new_key = f["kind"], f["frequency"], f["start"], f["constrained"]
        if key != new_key:
            key = new_key
            kind, frequency, start, constrained = key
            system = om.System()
            for mass in [1, 2, 3, 4, 0]:
                system.add_particle(mass)
            if constrained:
                system.add_constraint(0, 1, 1)
                system.add_constraint(2, 3, 1)
            external = om.CustomExternalForce("-k*x+0.1*y^2")
            external.add_per_particle_parameter("k")
            for i in range(5):
                external.add_particle(i, [0.2*(i+1)])
            external.set_force_group(2)
            system.add_force(external)
            remover = om.CMMotionRemover(frequency)
            assert remover.get_frequency() == frequency
            assert not remover.uses_periodic_boundary_conditions()
            remover.set_name("momentum probe")
            assert remover.get_name() == "momentum probe"
            remover.set_force_group(31)
            system.add_force(remover)
            integrator = [
                lambda: om.VerletIntegrator(0.0003),
                lambda: om.LangevinMiddleIntegrator(0, 0.7, 0.0003),
                lambda: om.BrownianIntegrator(0, 1.3, 0.0003),
            ][kind]()
            integrator.set_constraint_tolerance(1e-9)
            integrator.set_integration_force_groups(1 << 2)
            context = system.create_context(integrator)
        if f["steps"] == 0:
            if f["phase"]:
                remover.set_frequency(frequency+11)
                assert remover.get_frequency() == frequency+11
            context.set_positions([(0,0,0), (1,0,0), (0,1,0), (0,2,0), (0.5,0.3,0.9)])
            context.set_velocities([(0.3,-0.2,0.1), (-0.1,0.4,0.2), (0.2,0.1,-0.3), (0.1,-0.3,0.4), (0.7,0.8,0.9)])
            context.apply_velocity_constraints(1e-9)
            context.set_step_count(start)
            context.set_time(9.7)
            completed = 0
        context.step(f["steps"]-completed)
        completed = f["steps"]
        state = context.get_state(energy=True, forces=True, positions=True, velocities=True)
        assert state.step_count == f["step_count"]
        excluded = context.get_state(energy=True, forces=True, groups=1 << 31)
        assert excluded.potential_energy == 0 and all(v == 0 for p in excluded.forces for v in p)
        pairs = [("energy", state.potential_energy, f["energy"]),
                 ("kinetic", state.kinetic_energy, f["kinetic"]),
                 ("time", state.time, f["time"])]
        for metric, attribute in [("force", "forces"), ("position", "positions"), ("velocity", "velocities")]:
            pairs += [(metric, got, want) for a,b in zip(getattr(state, attribute), f[attribute])
                      for got,want in zip(a,b)]
        for metric, got, want in pairs:
            assert math.isfinite(got) and math.isclose(got, want, abs_tol=3e-8, rel_tol=3e-8), (
                key, f["phase"], f["steps"], metric, got, want)
            errors[metric].append(abs(got-want))
    print(json.dumps({"status": "passed", "fixtures": len(fixtures),
                      "comparisons": sum(map(len, errors.values())),
                      "max_absolute_error": {key: max(values) for key,values in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
