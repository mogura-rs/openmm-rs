"""Compare isotropic pressure, schedules and ideal-gas volume distributions."""
import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path


def pressure_system(om, rigid, variant, mask):
    system = om.System()
    for mass in [1, 3, 2, 0, 0, 4]:
        system.add_particle(mass)
    system.set_default_periodic_box_vectors((3, 0, 0), (0.4, 3.3, 0), (-0.2, 0.3, 3.5))
    bond = om.HarmonicBondForce()
    bond.add_bond(0, 1, 0.55, 17)
    bond.add_bond(2, 5, 0.7, 23)
    bond.set_uses_periodic_boundary_conditions(True)
    bond.set_force_group(1)
    system.add_force(bond)
    external = om.CustomExternalForce("k*(x*x+0.7*y*y+0.3*z*z)")
    external.add_global_parameter("k", 0.4)
    for i in range(6):
        external.add_particle(i, [])
    external.set_force_group(2)
    system.add_force(external)
    volume = om.CustomVolumeForce("0.03*v+0.07*ax*by")
    volume.set_force_group(5)
    system.add_force(volume)
    if variant >= 1:
        system.add_constraint(0, 1, 0.7)
    if variant == 2:
        system.set_virtual_site(4, om.TwoParticleAverageSite(0, 1, 0.3, 0.7))
        nb = om.NonbondedForce()
        nb.set_nonbonded_method("PME")
        nb.set_cutoff_distance(0.9)
        nb.set_pme_parameters(3.2, 16, 18, 20)
        for i in range(6):
            nb.add_particle((-1 if i % 2 else 1)*0.08, 0.2, 0.01)
        nb.add_exception(0, 1, 0, 0.2, 0)
        nb.set_force_group(3)
        nb.set_reciprocal_space_force_group(4)
        system.add_force(nb)
    barostat = om.MonteCarloBarostat(1.3, 300, 0, rigid)
    barostat.setRandomNumberSeed(17)
    barostat.setForceGroup(31)
    system.add_force(barostat)
    integrator = om.VerletIntegrator(0.00002)
    integrator.setConstraintTolerance(1e-10)
    integrator.setIntegrationForceGroups(mask)
    return system.create_context(integrator), barostat


def schedule_system(om, rigid):
    system = om.System()
    system.add_particle(1)
    system.add_particle(3)
    bond = om.HarmonicBondForce()
    bond.add_bond(0, 1, 0.7, 0)
    bond.set_uses_periodic_boundary_conditions(True)
    system.add_force(bond)
    barostat = om.MonteCarloBarostat(0, 0, 0, rigid)
    barostat.setRandomNumberSeed(73)
    system.add_force(barostat)
    integrator = om.VerletIntegrator(0.001)
    integrator.setIntegrationForceGroups(0)
    context = system.create_context(integrator)
    context.set_positions([(0.2, 0.3, 0.4), (0.9, 0.3, 0.4)])
    return context, barostat


def ideal_gas(om, rigid):
    kt = 1.380649e-23*6.02214076e23/1000*300
    scaled = 64 if rigid else 128
    length = ((scaled+1)*kt/(20*0.0602214076+0.2))**(1/3)
    system = om.System()
    for _ in range(128):
        system.add_particle(1)
    system.set_default_periodic_box_vectors((length, 0, 0), (0, length, 0), (0, 0, length))
    bond = om.HarmonicBondForce()
    for i in range(0, 128, 2):
        bond.add_bond(i, i+1, 0.02, 0)
    bond.set_uses_periodic_boundary_conditions(True)
    system.add_force(bond)
    bias = om.CustomVolumeForce("q*v")
    bias.add_global_parameter("q", 0.2)
    system.add_force(bias)
    barostat = om.MonteCarloBarostat(20, 300, 1, rigid)
    barostat.setRandomNumberSeed(91)
    system.add_force(barostat)
    context = system.create_context(om.VerletIntegrator(0.001))
    context.set_positions([(0.1+0.02*i, 0.2, 0.3) for i in range(128)])
    context.step(10000)
    blocks = []
    for _ in range(40):
        samples = []
        for _ in range(250):
            context.step(20)
            a, b, c = context.get_periodic_box_vectors()
            samples.append(a[0]*b[1]*c[2])
        blocks.append([statistics.mean(samples), statistics.mean(v*v for v in samples)])
    return blocks


def check_distribution(blocks, rigid):
    shape = (64 if rigid else 128)+1
    scale = (1.380649e-23*6.02214076e23/1000*300)/(20*0.0602214076+0.2)
    expected = [shape*scale, shape*(shape+1)*scale*scale]
    output = []
    for index, target in enumerate(expected):
        values = [block[index] for block in blocks]
        mean = statistics.mean(values)
        standard_error = statistics.stdev(values)/math.sqrt(len(values))
        assert abs(mean-target) < 6*standard_error, (rigid, index, mean, target, standard_error)
        output.append({"mean": mean, "expected": target, "block_standard_error": standard_error})
    return output


def probes(om):
    system = om.System()
    system.add_particle(1)
    barostat = om.MonteCarloBarostat(1, 300)
    system.add_force(barostat)
    integrator = om.VerletIntegrator(0.001)
    try:
        system.create_context(integrator)
        raise AssertionError("non-periodic barostat accepted")
    except RuntimeError:
        pass
    assert barostat.Pressure() == "MonteCarloPressure"
    assert barostat.Temperature() == "MonteCarloTemperature"
    assert not barostat.usesPeriodicBoundaryConditions()
    for rigid in [False, True]:
        context, force = schedule_system(om, rigid)
        force.set_scale_molecules_as_rigid(not rigid)
        force.setFrequency(1)
        force.setName("pressure probe")
        assert force.getName() == "pressure probe"
        before = context.get_state(positions=True)
        old_box = context.get_periodic_box_vectors()
        context.step(1)
        after = context.get_state(positions=True)
        scale = context.get_periodic_box_vectors()[0][0]/old_box[0][0]
        expected = 1 if rigid else scale
        assert math.isclose(math.dist(*after.positions)/math.dist(*before.positions), expected, abs_tol=1e-13)
        try:
            barostat.computeCurrentPressure(context)
            raise AssertionError("unrelated barostat accepted")
        except RuntimeError:
            pass
        context.get_system().remove_force(1)
        assert math.isfinite(force.compute_current_pressure(context))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(fixtures) == 386
    key = None
    errors = {name: [] for name in ["pressure", "energy", "kinetic", "positions", "velocities", "forces"]}
    statistical = []
    schedules = 0
    for row in fixtures:
        rigid = row["rigid"]
        if row["case"] == "ideal":
            statistical.append({"rigid": rigid, "reference": check_distribution(row["blocks"], rigid),
                                "rust": check_distribution(ideal_gas(om, rigid), rigid)})
        elif row["case"] == "schedule":
            step = row["step"]
            if step == 0:
                context, barostat = schedule_system(om, rigid)
            if step in [5, 13, 19]:
                barostat.setFrequency({5: 4, 13: 0, 19: 3}[step])
            context.set_step_count(step % 3)
            before = context.get_state(positions=True)
            old_length = context.get_periodic_box_vectors()[0][0]
            context.step(1)
            scale = context.get_periodic_box_vectors()[0][0]/old_length
            after = context.get_state(positions=True)
            assert (scale != 1) == row["changed"], (rigid, step, scale, row)
            assert math.isclose(math.dist(*after.positions)/math.dist(*before.positions), 1 if rigid else scale, abs_tol=1e-13)
            schedules += 1
        else:
            new_key = rigid, row["variant"], row["mask"]
            if new_key != key:
                key = new_key
                context, barostat = pressure_system(om, *key)
            phase = row["phase"]
            if row["steps"] == 0:
                scale = 1+phase*0.1
                context.set_periodic_box_vectors(*[tuple(v*scale for v in vector) for vector in [(3, 0, 0), (0.4, 3.3, 0), (-0.2, 0.3, 3.5)]])
                context.set_positions([(0.2,0.3,0.4), (0.9,0.3,0.4), (1.3,1.1,0.7), (0.5,1.7,1.3), (1.4,0.8,1.5), (1.9,1.2,0.8)])
                context.set_velocities([(0.3,-0.2,0.1), (-0.1,0.4,0.2), (0.2,0.1,-0.3), (0.7,0.8,0.9), (-0.3,0.6,0.5), (0.1,-0.3,0.4)])
                context.compute_virtual_sites()
                context.apply_velocity_constraints(1e-10)
                context.set_time(4.7)
                context.set_step_count(19)
                context.set_parameter("k", 0.4+phase*0.1)
                context.set_parameter(barostat.Pressure(), 2+phase)
                context.set_parameter(barostat.Temperature(), 260+10*phase)
                barostat.setDefaultPressure(11+phase)
                barostat.setDefaultTemperature(410+phase)
                done = 0
            context.step(row["steps"]-done)
            done = row["steps"]
            before = context.get_state(positions=True, velocities=True, forces=True, energy=True, parameters=True)
            pressure = barostat.computeCurrentPressure(context)
            after = context.get_state(positions=True, velocities=True)
            assert before.positions == after.positions and before.velocities == after.velocities
            assert before.periodic_box_vectors == after.periodic_box_vectors and before.time == after.time
            assert before.step_count == after.step_count == 19+done
            assert before.parameters[barostat.Pressure()] == 2+phase
            assert before.parameters[barostat.Temperature()] == 260+10*phase
            pairs = [("pressure", pressure, row["pressure"]),
                     ("energy", before.potential_energy, row["energy"]),
                     ("kinetic", before.kinetic_energy, row["kinetic"])]
            for name in ["positions", "velocities", "forces"]:
                pairs += [(name, a, b) for actual, expected in zip(getattr(before, name), row[name]) for a, b in zip(actual, expected)]
            for name, actual, expected in pairs:
                assert math.isfinite(actual) and math.isclose(actual, expected, abs_tol=3e-8, rel_tol=3e-8), (key, phase, done, name, actual, expected)
                errors[name].append(abs(actual-expected))
    probes(om)
    print(json.dumps({"status": "passed", "pressure_states": 324, "schedule_states": schedules,
                      "comparisons": sum(map(len, errors.values()))+schedules,
                      "max_absolute_error": {k: max(v) for k, v in errors.items()},
                      "ideal_gas_moments": statistical}, indent=2))


if __name__ == "__main__":
    main()
