"""Compare deterministic Andersen thermostat trajectories with OpenMM."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def statistics(om, reference):
    groups = 12000
    count = 3 * groups
    results = {}
    for constrained in [True, False]:
        system = om.System()
        for i in range(count):
            system.add_particle(1 if constrained else (i % 3 + 1)**2)
        if constrained:
            for i in range(groups):
                system.add_constraint(i, i+groups, 1)
                system.add_constraint(i+groups, i+2*groups, 1)
        thermostat = om.AndersenThermostat(0 if constrained else 310, 37 if constrained else 1e20)
        thermostat.set_random_number_seed(3721)
        system.add_force(thermostat)
        integrator = om.VerletIntegrator(0.01)
        integrator.set_constraint_tolerance(1e-9)
        context = system.create_context(integrator)
        context.set_positions([(i//groups if constrained else 0, 0, 0) for i in range(count)])
        context.set_velocities([(1,1,1)] * count)
        context.step(1)
        velocities = context.get_state(velocities=True).velocities
        if constrained:
            collided = [abs(v[0]) < 1e-9 for v in velocities[:groups]]
            for i, hit in enumerate(collided):
                for atom in [i, i+groups, i+2*groups]:
                    assert all(abs(v-(0 if hit else 1)) < 1e-8 for v in velocities[atom])
            probability = 1-math.exp(-0.37)
            tolerance = 6 * math.sqrt(probability*(1-probability)/groups)
            rate = sum(collided)/groups
            assert abs(rate-probability) < tolerance, rate
            assert abs(reference['collision']['rate']-probability) < tolerance
            results['collision_rate'] = {'rust': rate, 'reference': reference['collision']['rate'],
                                         'expected': probability, 'six_sigma_tolerance': tolerance}
        else:
            moments = []
            for mass in range(3):
                scale = math.sqrt((1.380649e-23*6.02214076e23/1000)*310/(mass+1)**2)
                values = [v/scale for i in range(mass,count,3) for v in velocities[i]]
                moments.append([sum(v**power for v in values)/len(values) for power in [1,2,4]])
            for dataset in [moments, reference['maxwell']['moments']]:
                for row in dataset:
                    for value, expected, tolerance in zip(row, [0,1,3], [0.035,0.06,0.3]):
                        assert abs(value-expected) < tolerance, (row, expected)
            results['normalized_maxwell_moments'] = {'rust': moments, 'reference': reference['maxwell']['moments']}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    reference_statistics = {f['statistics']: f for f in fixtures if 'statistics' in f}
    fixtures = [f for f in fixtures if 'statistics' not in f]
    assert len(fixtures) == 216
    statistical_results = statistics(om, reference_statistics)
    errors = {name: [] for name in ["energy", "kinetic", "time", "force", "position", "velocity"]}
    key = None
    for f in fixtures:
        new_key = f["kind"], f["mode"], f["constrained"]
        if key != new_key:
            key = new_key
            kind, mode, constrained = key
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
            remover = om.AndersenThermostat(300 if mode == 0 else 0, 1e20 if mode == 1 else 0)
            remover.set_random_number_seed(123)
            assert remover.get_random_number_seed() == 123
            assert remover.get_default_temperature() == (300 if mode == 0 else 0)
            assert remover.get_default_collision_frequency() == (1e20 if mode == 1 else 0)
            assert not remover.uses_periodic_boundary_conditions()
            remover.set_name("collision probe")
            assert remover.get_name() == "collision probe"
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
            if f["phase"] == 1:
                remover.set_default_temperature(777)
                remover.set_default_collision_frequency(88)
            if f["phase"] == 2:
                context.set_parameter(om.AndersenThermostat.temperature(), 0)
                context.set_parameter(om.AndersenThermostat.collision_frequency(), 0 if mode == 1 else 1e20)
            context.set_positions([(0,0,0), (1,0,0), (0,1,0), (0,2,0), (0.5,0.3,0.9)])
            context.set_velocities([(0.3,-0.2,0.1), (-0.1,0.4,0.2), (0.2,0.1,-0.3), (0.1,-0.3,0.4), (0.7,0.8,0.9)])
            context.apply_velocity_constraints(1e-9)
            context.set_step_count(17)
            context.set_time(9.7)
            completed = 0
        context.step(f["steps"]-completed)
        completed = f["steps"]
        state = context.get_state(energy=True, forces=True, positions=True, velocities=True)
        assert state.step_count == f["step_count"]
        assert context.get_parameter(om.AndersenThermostat.temperature()) == f["temperature"]
        assert context.get_parameter(om.AndersenThermostat.collision_frequency()) == f["frequency"]
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
    print(json.dumps({"status": "passed", "fixtures": len(fixtures), "statistics": statistical_results,
                      "comparisons": sum(map(len, errors.values())),
                      "max_absolute_error": {key: max(values) for key,values in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
