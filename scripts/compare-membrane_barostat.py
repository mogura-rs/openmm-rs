"""Compare membrane pressure, schedules and ideal-gas volume distributions."""
import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path


def pressure_system(om, rigid, variant, mask, triclinic):
    system = om.System()
    for mass in [1, 3, 2, 0, 0, 4]:
        system.add_particle(mass)
    system.set_default_periodic_box_vectors((3, 0, 0), (0.4 if triclinic else 0, 3.3, 0), (-0.2 if triclinic else 0, 0.3 if triclinic else 0, 3.5))
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
    barostat = om.MonteCarloMembraneBarostat(1.3, 0.7, 300, 0, 0, 0, rigid)
    barostat.setRandomNumberSeed(17)
    barostat.setForceGroup(31)
    system.add_force(barostat)
    integrator = om.VerletIntegrator(0.00002)
    integrator.setConstraintTolerance(1e-10)
    integrator.setIntegrationForceGroups(mask)
    return system.create_context(integrator), barostat


def schedule_system(om, rigid, xy, z):
    system = om.System()
    system.add_particle(1)
    system.add_particle(3)
    bond = om.HarmonicBondForce()
    bond.add_bond(0, 1, 0.7, 0)
    bond.set_uses_periodic_boundary_conditions(True)
    system.add_force(bond)
    barostat = om.MonteCarloMembraneBarostat(0, 0, 0, xy, z, 0, rigid)
    barostat.setRandomNumberSeed(73)
    system.add_force(barostat)
    integrator = om.VerletIntegrator(0.001)
    integrator.setIntegrationForceGroups(0)
    context = system.create_context(integrator)
    context.set_positions([(0.2, 0.3, 0.4), (0.9, 0.3, 0.4)])
    return context, barostat


def moments(om, rigid, xy, z):
    kt = 1.380649e-23*6.02214076e23/1000*300
    shape = (64 if rigid else 128)+1
    x = math.sqrt(shape*kt/10) if xy == 0 else shape*kt/50
    y = x if xy == 0 else shape*kt/60
    height = shape*kt/70
    if z == 2:
        x = y = height = 3
    system = om.System()
    for _ in range(128):
        system.add_particle(1)
    system.set_default_periodic_box_vectors((x,0,0), (0,y,0), (0,0,height))
    bond = om.HarmonicBondForce()
    for i in range(0,128,2):
        bond.add_bond(i,i+1,0.02,0)
    bond.set_uses_periodic_boundary_conditions(True)
    system.add_force(bond)
    bias = "10*ax*by+70*cz" if xy == 0 else "50*ax+60*by+70*cz"
    if z == 2:
        bias = "10*(ax*by-9)^2" if xy == 0 else "50*((ax-3)^2+(by-3)^2)"
    potential = om.CustomVolumeForce("-pc*v+gc*ax*by+"+bias)
    potential.add_global_parameter("pc",20*0.0602214076)
    potential.add_global_parameter("gc",7*0.0602214076)
    system.add_force(potential)
    barostat = om.MonteCarloMembraneBarostat(5,2,150,xy,z,1,rigid)
    barostat.setRandomNumberSeed(91)
    system.add_force(barostat)
    context = system.create_context(om.VerletIntegrator(0.001))
    context.set_positions([(0.1+0.02*i,0.2,0.3) for i in range(128)])
    context.set_parameter(barostat.Pressure(),20)
    context.set_parameter(barostat.SurfaceTension(),7)
    context.set_parameter(barostat.Temperature(),300)
    barostat.setDefaultPressure(99)
    barostat.setDefaultSurfaceTension(19)
    barostat.setDefaultTemperature(50)
    context.step(10000)
    blocks = []
    for _ in range(40):
        samples = []
        for _ in range(250):
            context.step(20)
            a,b,c = context.get_periodic_box_vectors()
            if z == 2:
                assert math.isclose(a[0]*b[1]*c[2],27,abs_tol=1e-8)
            if z == 1:
                assert c[2] == height
            values = [a[0]*b[1]] if xy == 0 else [a[0],b[1]]
            if z != 1:
                values.append(c[2])
            samples.append([moment for v in values for moment in [v,v*v]])
        blocks.append([statistics.mean(column) for column in zip(*samples)])
    return blocks


def check_moments(reference, rust, rigid, xy, z):
    shape = (64 if rigid else 128)+1
    kt = 1.380649e-23*6.02214076e23/1000*300
    coefficients = [10] if xy == 0 else [50,60]
    if z != 1:
        coefficients.append(70)
    expected = [value for coefficient in coefficients for value in [shape*kt/coefficient,shape*(shape+1)*(kt/coefficient)**2]]
    results = []
    for index, (source, actual) in enumerate(zip(zip(*reference), zip(*rust))):
        source_mean, rust_mean = statistics.mean(source), statistics.mean(actual)
        source_se = statistics.stdev(source)/math.sqrt(len(source))
        rust_se = statistics.stdev(actual)/math.sqrt(len(actual))
        if z == 2:
            assert abs(source_mean-rust_mean) < 6*math.hypot(source_se,rust_se), (rigid,xy,z,index,source_mean,rust_mean,source_se,rust_se)
        else:
            for mean,se in [(source_mean,source_se),(rust_mean,rust_se)]:
                assert abs(mean-expected[index]) < 6*se, (rigid,xy,z,index,mean,expected[index],se)
        results.append({"reference_mean":source_mean,"rust_mean":rust_mean,"reference_standard_error":source_se,"rust_standard_error":rust_se,
                        "expected":expected[index] if z != 2 else None})
    return results


def probes(om):
    cls=om.MonteCarloMembraneBarostat
    assert [cls.XYIsotropic,cls.XYAnisotropic,cls.ZFree,cls.ZFixed,cls.ConstantVolume] == [0,1,0,1,2]
    assert [cls.Pressure(),cls.SurfaceTension(),cls.Temperature()] == ["MembraneMonteCarloPressure","MembraneMonteCarloSurfaceTension","MembraneMonteCarloTemperature"]
    unrelated = cls(1,2,300,0,0)
    assert not unrelated.usesPeriodicBoundaryConditions()
    for rigid in [False,True]:
        context, barostat = schedule_system(om,rigid,0,1)
        barostat.setScaleMoleculesAsRigid(not rigid)
        barostat.setFrequency(1)
        before = context.get_state(positions=True)
        old_box = context.get_periodic_box_vectors()
        context.step(1)
        after = context.get_state(positions=True)
        scale=context.get_periodic_box_vectors()[0][0]/old_box[0][0]
        assert math.isclose(math.dist(*after.positions)/math.dist(*before.positions),1 if rigid else scale,abs_tol=1e-13)
        try:
            unrelated.computeCurrentPressure(context)
            raise AssertionError("unrelated barostat accepted")
        except RuntimeError:
            pass
        context.get_system().remove_force(1)
        assert all(map(math.isfinite,barostat.computeCurrentPressure(context)))
    for xy,z in [(2,0),(0,3)]:
        try:
            cls(1,2,300,xy,z)
            raise AssertionError("invalid mode accepted")
        except RuntimeError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures = [json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(fixtures) == 1020
    key = None
    errors = {name: [] for name in ["pressure", "energy", "kinetic", "positions", "velocities", "forces"]}
    statistical = []
    schedules = 0
    for row in fixtures:
        rigid = row["rigid"]
        if row["case"] == "moments":
            xy,z=row["xy"],row["z"]
            statistical.append({"rigid":rigid,"xy":xy,"z":z,"moments":check_moments(row["blocks"],moments(om,rigid,xy,z),rigid,xy,z)})
        elif row["case"] == "schedule":
            step = row["step"]
            if step == 0:
                context, barostat = schedule_system(om, rigid, row["initial_xy"], row["initial_z"])
            if step in [7,17]:
                barostat.setXYMode(1-barostat.getXYMode())
            if step in [11,23]:
                barostat.setZMode((barostat.getZMode()+1)%3)
            assert barostat.getXYMode() == row["xy"] and barostat.getZMode() == row["z"]
            if step in [5, 13, 19]:
                barostat.setFrequency({5: 4, 13: 0, 19: 3}[step])
            context.set_step_count(step % 3)
            before = context.get_state(positions=True)
            old_box = context.get_periodic_box_vectors()
            context.step(1)
            new_box = context.get_periodic_box_vectors()
            scales = [new_box[axis][axis]/old_box[axis][axis] for axis in range(3)]
            after = context.get_state(positions=True)
            assert any(scale != 1 for scale in scales) == row["changed"], row
            if row["xy"] == 0:
                assert math.isclose(scales[0],scales[1],abs_tol=1e-14)
            else:
                assert scales[0] == 1 or scales[1] == 1
            if row["z"] == 1:
                assert scales[2] == 1
            if row["z"] == 2:
                assert math.isclose(math.prod(scales),1,abs_tol=1e-14)
            for axis, scale in enumerate(scales):
                for atom in range(2):
                    center = sum(p[axis] for p in before.positions)/2 if rigid else before.positions[atom][axis]
                    expected = before.positions[atom][axis]+(scale-1)*center
                    assert math.isclose(after.positions[atom][axis], expected, abs_tol=1e-13)
            schedules += 1
        else:
            new_key = rigid, row["variant"], row["mask"], row["triclinic"]
            if new_key != key:
                key = new_key
                context, barostat = pressure_system(om, *key)
            phase = row["phase"]
            if row["steps"] == 0:
                triclinic = row["triclinic"]
                scale = 1+phase*0.1
                context.set_periodic_box_vectors(*[tuple(v*scale for v in vector) for vector in [(3, 0, 0), (0.4 if triclinic else 0, 3.3, 0), (-0.2 if triclinic else 0, 0.3 if triclinic else 0, 3.5)]])
                context.set_positions([(0.2,0.3,0.4), (0.9,0.3,0.4), (1.3,1.1,0.7), (0.5,1.7,1.3), (1.4,0.8,1.5), (1.9,1.2,0.8)])
                context.set_velocities([(0.3,-0.2,0.1), (-0.1,0.4,0.2), (0.2,0.1,-0.3), (0.7,0.8,0.9), (-0.3,0.6,0.5), (0.1,-0.3,0.4)])
                context.compute_virtual_sites()
                context.apply_velocity_constraints(1e-10)
                context.set_time(4.7)
                context.set_step_count(19)
                context.set_parameter("k", 0.4+phase*0.1)
                context.set_parameter(barostat.Pressure(), 2+phase)
                context.set_parameter(barostat.SurfaceTension(), 3+phase)
                context.set_parameter(barostat.Temperature(), 260+10*phase)
                barostat.setDefaultPressure(11+phase)
                barostat.setDefaultSurfaceTension(12+phase)
                barostat.setDefaultTemperature(410+phase)
                done = 0
            barostat.setXYMode((phase+row["steps"])%2)
            barostat.setZMode(phase)
            context.step(row["steps"]-done)
            done = row["steps"]
            before = context.get_state(positions=True, velocities=True, forces=True, energy=True, parameters=True)
            pressure = barostat.computeCurrentPressure(context)
            after = context.get_state(positions=True, velocities=True)
            assert before.positions == after.positions and before.velocities == after.velocities
            assert before.periodic_box_vectors == after.periodic_box_vectors and before.time == after.time
            assert before.step_count == after.step_count == 19+done
            assert before.parameters[barostat.Pressure()] == 2+phase
            assert before.parameters[barostat.SurfaceTension()] == 3+phase
            assert before.parameters[barostat.Temperature()] == 260+10*phase
            pairs = [("pressure", a, b) for a, b in zip(pressure, row["pressure"])]
            pairs += [("energy", before.potential_energy, row["energy"]),
                     ("kinetic", before.kinetic_energy, row["kinetic"])]
            for name in ["positions", "velocities", "forces"]:
                pairs += [(name, a, b) for actual, expected in zip(getattr(before, name), row[name]) for a, b in zip(actual, expected)]
            for name, actual, expected in pairs:
                assert math.isfinite(actual) and math.isclose(actual, expected, abs_tol=3e-8, rel_tol=3e-8), (key, phase, done, name, actual, expected)
                errors[name].append(abs(actual-expected))
    probes(om)
    print(json.dumps({"status": "passed", "pressure_states": 648, "schedule_states": schedules,
                      "comparisons": sum(map(len, errors.values()))+schedules,
                      "max_absolute_error": {k: max(v) for k, v in errors.items()},
                      "box_moments": statistical}, indent=2))


if __name__ == "__main__":
    main()
